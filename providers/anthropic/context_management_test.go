package anthropic

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"net/http"
	"net/http/httptest"
	"reflect"
	"slices"
	"strings"
	"testing"
	"time"

	"charm.land/fantasy"
	"github.com/stretchr/testify/require"
)

// wantContextManagementBeta is written out here instead of taken from the
// package. So a wrong flag in the package fails these tests.
const wantContextManagementBeta = "context-management-2025-06-27"

type capturedRequest struct {
	header http.Header
	query  string
	body   map[string]any
}

// newCapturingServer answers like the Messages API, as JSON or an event
// stream. It hands back each request it saw.
func newCapturingServer(stream bool) (*httptest.Server, <-chan capturedRequest) {
	calls := make(chan capturedRequest, 4)
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		var body map[string]any
		if r.Body != nil {
			_ = json.NewDecoder(r.Body).Decode(&body)
		}
		calls <- capturedRequest{header: r.Header.Clone(), query: r.URL.RawQuery, body: body}
		if !stream {
			w.Header().Set("Content-Type", "application/json")
			_ = json.NewEncoder(w).Encode(mockAnthropicGenerateResponse())
			return
		}
		w.Header().Set("Content-Type", "text/event-stream")
		w.WriteHeader(http.StatusOK)
		for _, chunk := range events(
			map[string]any{"type": "message_start", "message": map[string]any{}},
			map[string]any{"type": "message_stop"},
		) {
			_, _ = w.Write([]byte(chunk))
			if flusher, ok := w.(http.Flusher); ok {
				flusher.Flush()
			}
		}
	}))
	return server, calls
}

func newTestModel(t *testing.T, url string) fantasy.LanguageModel {
	t.Helper()
	provider, err := New(WithAPIKey("test-api-key"), WithBaseURL(url))
	require.NoError(t, err)
	model, err := provider.LanguageModel(context.Background(), "claude-sonnet-4-20250514")
	require.NoError(t, err)
	return model
}

func awaitCaptured(t *testing.T, calls <-chan capturedRequest) capturedRequest {
	t.Helper()
	select {
	case call := <-calls:
		return call
	case <-time.After(2 * time.Second):
		t.Fatal("timed out waiting for Anthropic request")
		return capturedRequest{}
	}
}

// sentContextManagement reports whether a request carried want as its
// context_management, with the beta header and query that turn it on.
func sentContextManagement(call capturedRequest, want map[string]any) error {
	got, ok := call.body["context_management"]
	if !ok {
		return errors.New("no context_management in the body")
	}
	if !reflect.DeepEqual(got, want) {
		return fmt.Errorf("context_management is %v, want %v", got, want)
	}
	if !slices.Contains(call.header.Values("Anthropic-Beta"), wantContextManagementBeta) {
		return fmt.Errorf("no %s in Anthropic-Beta %v", wantContextManagementBeta, call.header.Values("Anthropic-Beta"))
	}
	if !strings.Contains(call.query, "beta=true") {
		return fmt.Errorf("no beta=true in the query %q", call.query)
	}
	return nil
}

// sentNoContextManagement reports whether a request left out context
// management entirely. It checks the body field, the beta header and the
// beta query.
func sentNoContextManagement(call capturedRequest) error {
	if _, ok := call.body["context_management"]; ok {
		return errors.New("context_management in the body")
	}
	if slices.Contains(call.header.Values("Anthropic-Beta"), wantContextManagementBeta) {
		return fmt.Errorf("%s in Anthropic-Beta", wantContextManagementBeta)
	}
	if strings.Contains(call.query, "beta=true") {
		return fmt.Errorf("beta=true in the query %q", call.query)
	}
	return nil
}

// bothEdits is a context management that uses every field of both edits.
func bothEdits() *ContextManagement {
	clearInputs := true
	return &ContextManagement{Edits: []ContextEdit{
		{
			Type: ContextEditClearThinking,
			Keep: &ContextAmount{Type: ContextAmountThinkingTurns, Value: 2},
		},
		{
			Type:            ContextEditClearToolUses,
			Trigger:         &ContextAmount{Type: ContextAmountInputTokens, Value: 100000},
			Keep:            &ContextAmount{Type: ContextAmountToolUses, Value: 3},
			ClearAtLeast:    &ContextAmount{Type: ContextAmountInputTokens, Value: 5000},
			ExcludeTools:    []string{"memory"},
			ClearToolInputs: &clearInputs,
		},
	}}
}

// bothEditsWire is bothEdits as the API takes it.
func bothEditsWire() map[string]any {
	return map[string]any{
		"edits": []any{
			map[string]any{
				"type": "clear_thinking_20251015",
				"keep": map[string]any{"type": "thinking_turns", "value": float64(2)},
			},
			map[string]any{
				"type":              "clear_tool_uses_20250919",
				"trigger":           map[string]any{"type": "input_tokens", "value": float64(100000)},
				"keep":              map[string]any{"type": "tool_uses", "value": float64(3)},
				"clear_at_least":    map[string]any{"type": "input_tokens", "value": float64(5000)},
				"exclude_tools":     []any{"memory"},
				"clear_tool_inputs": true,
			},
		},
	}
}

func TestParseOptions_ContextManagement(t *testing.T) {
	t.Parallel()

	options, err := ParseOptions(map[string]any{
		"context_management": map[string]any{
			"edits": []any{
				map[string]any{
					"type": "clear_tool_uses_20250919",
					"keep": map[string]any{"type": "tool_uses", "value": 3},
				},
				map[string]any{
					"type": "clear_thinking_20251015",
					"keep": map[string]any{"type": "all"},
				},
			},
		},
	})
	require.NoError(t, err)
	require.NotNil(t, options.ContextManagement)
	require.Equal(t, []ContextEdit{
		{Type: ContextEditClearToolUses, Keep: &ContextAmount{Type: ContextAmountToolUses, Value: 3}},
		{Type: ContextEditClearThinking, Keep: &ContextAmount{Type: ContextAmountAll}},
	}, options.ContextManagement.Edits)
}

// Provider options read from a JSON config file hold their numbers as
// float64.
func TestParseOptions_ContextManagementFromJSONConfig(t *testing.T) {
	t.Parallel()

	var config map[string]any
	require.NoError(t, json.Unmarshal([]byte(`{"context_management": {"edits": [
		{"type": "clear_tool_uses_20250919",
		 "trigger": {"type": "input_tokens", "value": 100000},
		 "keep": {"type": "tool_uses", "value": 3},
		 "exclude_tools": ["memory"],
		 "clear_tool_inputs": true}
	]}}`), &config))

	options, err := ParseOptions(config)
	require.NoError(t, err)
	clearInputs := true
	require.Equal(t, []ContextEdit{{
		Type:            ContextEditClearToolUses,
		Trigger:         &ContextAmount{Type: ContextAmountInputTokens, Value: 100000},
		Keep:            &ContextAmount{Type: ContextAmountToolUses, Value: 3},
		ExcludeTools:    []string{"memory"},
		ClearToolInputs: &clearInputs,
	}}, options.ContextManagement.Edits)
}

// Options saved with a message come back through the provider registry. So
// this test sends them through the registry too.
func TestProviderOptions_ContextManagementRoundTrip(t *testing.T) {
	t.Parallel()

	for name, cm := range map[string]*ContextManagement{
		"both edits": bothEdits(),
		"keep all":   {Edits: []ContextEdit{{Type: ContextEditClearThinking, Keep: &ContextAmount{Type: ContextAmountAll}}}},
	} {
		t.Run(name, func(t *testing.T) {
			t.Parallel()
			data, err := json.Marshal(NewProviderOptions(&ProviderOptions{ContextManagement: cm}))
			require.NoError(t, err)
			var raw map[string]json.RawMessage
			require.NoError(t, json.Unmarshal(data, &raw))
			back, err := fantasy.UnmarshalProviderOptions(raw)
			require.NoError(t, err)
			options, ok := back[Name].(*ProviderOptions)
			require.True(t, ok)
			require.Equal(t, cm, options.ContextManagement)
		})
	}
}

func TestParseOptions_ContextManagementRefusesMalformed(t *testing.T) {
	t.Parallel()

	for name, cm := range map[string]any{
		"edits not a list":    map[string]any{"edits": "clear everything"},
		"an edit not a map":   map[string]any{"edits": []any{"clear_tool_uses_20250919"}},
		"a value not a count": map[string]any{"edits": []any{map[string]any{"type": "clear_tool_uses_20250919", "keep": map[string]any{"type": "tool_uses", "value": "three"}}}},
	} {
		t.Run(name, func(t *testing.T) {
			t.Parallel()
			_, err := ParseOptions(map[string]any{"context_management": cm})
			require.Error(t, err)
		})
	}
}

func TestContextAmount_MarshalJSON(t *testing.T) {
	t.Parallel()

	tests := []struct {
		name   string
		amount ContextAmount
		want   string
	}{
		{"all doesn't take a value", ContextAmount{Type: ContextAmountAll}, `{"type":"all"}`},
		{"all drops a stray value", ContextAmount{Type: ContextAmountAll, Value: 4}, `{"type":"all"}`},
		{"a count keeps its value", ContextAmount{Type: ContextAmountToolUses, Value: 3}, `{"type":"tool_uses","value":3}`},
		{"a zero count is still sent", ContextAmount{Type: ContextAmountThinkingTurns}, `{"type":"thinking_turns","value":0}`},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			t.Parallel()
			got, err := json.Marshal(tt.amount)
			require.NoError(t, err)
			require.JSONEq(t, tt.want, string(got))
		})
	}
}

func TestGenerate_SendsContextManagement(t *testing.T) {
	t.Parallel()

	server, calls := newCapturingServer(false)
	defer server.Close()

	_, err := newTestModel(t, server.URL).Generate(context.Background(), fantasy.Call{
		Prompt:          testPrompt(),
		ProviderOptions: NewProviderOptions(&ProviderOptions{ContextManagement: bothEdits()}),
	})
	require.NoError(t, err)
	require.NoError(t, sentContextManagement(awaitCaptured(t, calls), bothEditsWire()))
}

func TestStream_SendsContextManagement(t *testing.T) {
	t.Parallel()

	server, calls := newCapturingServer(true)
	defer server.Close()

	stream, err := newTestModel(t, server.URL).Stream(context.Background(), fantasy.Call{
		Prompt:          testPrompt(),
		ProviderOptions: NewProviderOptions(&ProviderOptions{ContextManagement: bothEdits()}),
	})
	require.NoError(t, err)
	stream(func(fantasy.StreamPart) bool { return true })
	require.NoError(t, sentContextManagement(awaitCaptured(t, calls), bothEditsWire()))
}

func TestGenerate_SendsOnlyTheContextEditFieldsSet(t *testing.T) {
	t.Parallel()

	server, calls := newCapturingServer(false)
	defer server.Close()

	_, err := newTestModel(t, server.URL).Generate(context.Background(), fantasy.Call{
		Prompt: testPrompt(),
		ProviderOptions: NewProviderOptions(&ProviderOptions{ContextManagement: &ContextManagement{
			Edits: []ContextEdit{{Type: ContextEditClearThinking, Keep: &ContextAmount{Type: ContextAmountAll}}},
		}}),
	})
	require.NoError(t, err)
	require.NoError(t, sentContextManagement(awaitCaptured(t, calls), map[string]any{
		"edits": []any{map[string]any{"type": "clear_thinking_20251015", "keep": map[string]any{"type": "all"}}},
	}))
}

func TestGenerate_NoContextManagementWithoutEdits(t *testing.T) {
	t.Parallel()

	tests := []struct {
		name    string
		options *ProviderOptions
	}{
		{"no provider options", nil},
		{"no context management", &ProviderOptions{}},
		{"context management without edits", &ProviderOptions{ContextManagement: &ContextManagement{}}},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			t.Parallel()

			server, calls := newCapturingServer(false)
			defer server.Close()

			call := fantasy.Call{Prompt: testPrompt()}
			if tt.options != nil {
				call.ProviderOptions = NewProviderOptions(tt.options)
			}
			_, err := newTestModel(t, server.URL).Generate(context.Background(), call)
			require.NoError(t, err)
			require.NoError(t, sentNoContextManagement(awaitCaptured(t, calls)))
		})
	}
}

func TestGenerate_ContextManagementBesideBetaTools(t *testing.T) {
	t.Parallel()

	server, calls := newCapturingServer(false)
	defer server.Close()

	cuTool := jsonRoundTripTool(t, NewComputerUseTool(ComputerUseToolOptions{
		DisplayWidthPx:  1920,
		DisplayHeightPx: 1080,
		ToolVersion:     ComputerUse20250124,
	}, noopComputerRun))

	_, err := newTestModel(t, server.URL).Generate(context.Background(), fantasy.Call{
		Prompt:          testPrompt(),
		Tools:           []fantasy.Tool{cuTool},
		ProviderOptions: NewProviderOptions(&ProviderOptions{ContextManagement: bothEdits()}),
	})
	require.NoError(t, err)

	call := awaitCaptured(t, calls)
	require.NoError(t, sentContextManagement(call, bothEditsWire()))
	require.Contains(t, call.header.Values("Anthropic-Beta"), "computer-use-2025-01-24")
	require.NotEmpty(t, call.body["tools"])
}

func TestGenerate_ExtraBodyOverridesContextManagement(t *testing.T) {
	t.Parallel()

	server, calls := newCapturingServer(false)
	defer server.Close()

	override := map[string]any{"edits": []any{map[string]any{"type": "clear_tool_uses_20250919"}}}
	_, err := newTestModel(t, server.URL).Generate(context.Background(), fantasy.Call{
		Prompt: testPrompt(),
		ProviderOptions: NewProviderOptions(&ProviderOptions{
			ContextManagement: bothEdits(),
			ExtraBody:         map[string]any{"context_management": override},
		}),
	})
	require.NoError(t, err)
	require.NoError(t, sentContextManagement(awaitCaptured(t, calls), override))
}

// The anti-tests hand each check above a request that is wrong in one way.
// The check must fail on it. So a test that uses the check can't pass
// without checking anything.

func sentRequest() capturedRequest {
	return capturedRequest{
		header: http.Header{"Anthropic-Beta": {wantContextManagementBeta}},
		query:  "beta=true",
		body:   map[string]any{"context_management": bothEditsWire()},
	}
}

func TestSentContextManagement_FailsOnABrokenRequest(t *testing.T) {
	t.Parallel()

	require.NoError(t, sentContextManagement(sentRequest(), bothEditsWire()))

	for name, breakIt := range map[string]func(*capturedRequest){
		"no field":    func(c *capturedRequest) { delete(c.body, "context_management") },
		"other edits": func(c *capturedRequest) { c.body["context_management"] = map[string]any{"edits": []any{}} },
		"a value changed": func(c *capturedRequest) {
			c.body["context_management"].(map[string]any)["edits"].([]any)[1].(map[string]any)["keep"] = map[string]any{"type": "tool_uses", "value": float64(4)}
		},
		"no beta header":    func(c *capturedRequest) { c.header.Del("Anthropic-Beta") },
		"only another beta": func(c *capturedRequest) { c.header.Set("Anthropic-Beta", "computer-use-2025-01-24") },
		"no beta query":     func(c *capturedRequest) { c.query = "" },
	} {
		t.Run(name, func(t *testing.T) {
			t.Parallel()
			call := sentRequest()
			breakIt(&call)
			require.Error(t, sentContextManagement(call, bothEditsWire()))
		})
	}
}

func TestSentNoContextManagement_FailsOnAnySign(t *testing.T) {
	t.Parallel()

	require.NoError(t, sentNoContextManagement(capturedRequest{header: http.Header{}, body: map[string]any{}}))
	require.Error(t, sentNoContextManagement(sentRequest()))

	for name, call := range map[string]capturedRequest{
		"the field alone":  {header: http.Header{}, body: map[string]any{"context_management": bothEditsWire()}},
		"the header alone": {header: http.Header{"Anthropic-Beta": {wantContextManagementBeta}}, body: map[string]any{}},
		"the query alone":  {header: http.Header{}, query: "beta=true", body: map[string]any{}},
	} {
		t.Run(name, func(t *testing.T) {
			t.Parallel()
			require.Error(t, sentNoContextManagement(call))
		})
	}
}

// newReplyServer answers every request with reply as JSON. When chunks isn't
// empty, it answers with chunks as an event stream instead.
func newReplyServer(reply map[string]any, chunks []string) *httptest.Server {
	return httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if len(chunks) == 0 {
			w.Header().Set("Content-Type", "application/json")
			_ = json.NewEncoder(w).Encode(reply)
			return
		}
		w.Header().Set("Content-Type", "text/event-stream")
		w.WriteHeader(http.StatusOK)
		for _, chunk := range chunks {
			_, _ = w.Write([]byte(chunk))
			if flusher, ok := w.(http.Flusher); ok {
				flusher.Flush()
			}
		}
	}))
}

var twoEditsApplied = []any{
	map[string]any{"type": "clear_thinking_20251015", "cleared_thinking_turns": 2, "cleared_input_tokens": 900},
	map[string]any{"type": "clear_tool_uses_20250919", "cleared_tool_uses": 3, "cleared_input_tokens": 12000},
}

var twoEditsAppliedMetadata = &ContextManagementMetadata{AppliedEdits: []AppliedContextEdit{
	{Type: ContextEditClearThinking, ClearedThinkingTurns: 2, ClearedInputTokens: 900},
	{Type: ContextEditClearToolUses, ClearedToolUses: 3, ClearedInputTokens: 12000},
}}

func replyWith(contextManagement any) map[string]any {
	reply := mockAnthropicGenerateResponse()
	if contextManagement != nil {
		reply["context_management"] = contextManagement
	}
	return reply
}

// events is a stream of server-sent events, one for each payload. Each
// event is named after its payload's type.
func events(payloads ...map[string]any) []string {
	var chunks []string
	for _, payload := range payloads {
		data, err := json.Marshal(payload)
		if err != nil {
			panic(err)
		}
		chunks = append(chunks, "event: "+payload["type"].(string)+"\n", "data: "+string(data)+"\n\n")
	}
	return chunks
}

// replyStream is a streamed reply. The content events come after its
// message_start. Its message_delta is end.
func replyStream(end map[string]any, content ...map[string]any) []string {
	start := map[string]any{"type": "message_start", "message": map[string]any{
		"id": "msg_1", "type": "message", "role": "assistant", "content": []any{},
		"model": "claude-sonnet-4-20250514", "usage": map[string]any{"input_tokens": 10, "output_tokens": 0},
	}}
	all := append([]map[string]any{start}, content...)
	return events(append(all, end, map[string]any{"type": "message_stop"})...)
}

// messageDelta is a message_delta with stopReason. It carries
// contextManagement when that isn't nil.
func messageDelta(stopReason string, contextManagement any) map[string]any {
	delta := map[string]any{
		"type":  "message_delta",
		"delta": map[string]any{"stop_reason": stopReason, "stop_sequence": nil},
		"usage": map[string]any{"output_tokens": 1},
	}
	if contextManagement != nil {
		delta["context_management"] = contextManagement
	}
	return delta
}

// finishMetadata streams a call and returns the provider metadata of its
// finish part.
func finishMetadata(t *testing.T, url string) fantasy.ProviderMetadata {
	t.Helper()
	stream, err := newTestModel(t, url).Stream(context.Background(), fantasy.Call{
		Prompt:          testPrompt(),
		ProviderOptions: NewProviderOptions(&ProviderOptions{ContextManagement: bothEdits()}),
	})
	require.NoError(t, err)
	var finish *fantasy.StreamPart
	stream(func(part fantasy.StreamPart) bool {
		if part.Type == fantasy.StreamPartTypeFinish {
			finish = &part
		}
		return true
	})
	require.NotNil(t, finish, "the stream never finished")
	return finish.ProviderMetadata
}

func TestGenerate_ReportsAppliedEdits(t *testing.T) {
	t.Parallel()

	tests := []struct {
		name  string
		reply any
		want  *ContextManagementMetadata
	}{
		{"edits applied", map[string]any{"applied_edits": twoEditsApplied}, twoEditsAppliedMetadata},
		{"nothing needed clearing", map[string]any{"applied_edits": []any{}}, &ContextManagementMetadata{AppliedEdits: []AppliedContextEdit{}}},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			t.Parallel()
			server := newReplyServer(replyWith(tt.reply), nil)
			defer server.Close()

			response, err := newTestModel(t, server.URL).Generate(context.Background(), fantasy.Call{
				Prompt:          testPrompt(),
				ProviderOptions: NewProviderOptions(&ProviderOptions{ContextManagement: bothEdits()}),
			})
			require.NoError(t, err)
			require.Equal(t, tt.want, response.ProviderMetadata[Name])
		})
	}
}

func TestGenerate_NoMetadataWithoutContextManagement(t *testing.T) {
	t.Parallel()

	server := newReplyServer(replyWith(nil), nil)
	defer server.Close()

	response, err := newTestModel(t, server.URL).Generate(context.Background(), fantasy.Call{Prompt: testPrompt()})
	require.NoError(t, err)
	require.NotContains(t, response.ProviderMetadata, Name)
}

func TestStream_ReportsAppliedEdits(t *testing.T) {
	t.Parallel()

	server := newReplyServer(nil, replyStream(messageDelta("end_turn", map[string]any{"applied_edits": twoEditsApplied})))
	defer server.Close()

	require.Equal(t, twoEditsAppliedMetadata, finishMetadata(t, server.URL)[Name])
}

func TestStream_NoMetadataWithoutContextManagement(t *testing.T) {
	t.Parallel()

	server := newReplyServer(nil, replyStream(messageDelta("end_turn", nil)))
	defer server.Close()

	require.NotContains(t, finishMetadata(t, server.URL), Name)
}

func TestContextManagementMetadata_RoundTrip(t *testing.T) {
	t.Parallel()

	data, err := json.Marshal(fantasy.ProviderMetadata{Name: twoEditsAppliedMetadata})
	require.NoError(t, err)
	var raw map[string]json.RawMessage
	require.NoError(t, json.Unmarshal(data, &raw))
	back, err := fantasy.UnmarshalProviderMetadata(raw)
	require.NoError(t, err)
	require.Equal(t, twoEditsAppliedMetadata, back[Name])
}

func TestContextManagementMetadata_RefusesMalformed(t *testing.T) {
	t.Parallel()

	_, err := fantasy.UnmarshalProviderMetadata(map[string]json.RawMessage{
		Name: json.RawMessage(`{"type":"anthropic.context_management_metadata","data":"not an object"}`),
	})
	require.Error(t, err)
}

// Compaction. The shapes below are what the API sent for claude-haiku-5-5 on
// 2026-10-21. The block had only type and content. The stream was a
// content_block_start with null content, one compaction_delta and the stop.
// The tests also cover the fields the SDK documents that this model didn't
// send.

const compactionSummary = "The user pasted an inventory of 2,600 numbered crates and asked about item 37."

// compactionBlockAsSent is a block in the shape claude-haiku-5-5 sent.
var compactionBlockAsSent = `{"type":"compaction","content":"` + compactionSummary + `"}`

// compactionBlockDocumented has every field the SDK documents.
var compactionBlockDocumented = `{"type":"compaction","content":"` + compactionSummary + `","encrypted_content":"opaque-1","signature":"sig-1","tool_changes":[]}`

func compactionEdit() *ContextManagement {
	trigger := &ContextAmount{Type: ContextAmountInputTokens, Value: 50000}
	return &ContextManagement{Edits: []ContextEdit{{Type: ContextEditCompact, Trigger: trigger}}}
}

func compactionReply(block string, stopReason string) map[string]any {
	reply := mockAnthropicGenerateResponse()
	var b map[string]any
	_ = json.Unmarshal([]byte(block), &b)
	reply["content"] = []any{b, map[string]any{"type": "text", "text": "Item 37 is a green crate from Oslo."}}
	if stopReason == "compaction" {
		reply["content"] = []any{b}
	}
	reply["stop_reason"] = stopReason
	reply["context_management"] = map[string]any{"applied_edits": []any{}}
	reply["usage"] = map[string]any{
		"input_tokens": 214, "output_tokens": 394,
		"iterations": []any{
			map[string]any{"type": "compaction", "input_tokens": 135983, "output_tokens": 1024},
			map[string]any{"type": "message", "input_tokens": 214, "output_tokens": 394},
		},
	}
	return reply
}

func TestGenerate_CompactionBlockBecomesReasoning(t *testing.T) {
	t.Parallel()

	for name, block := range map[string]string{"as sent": compactionBlockAsSent, "documented": compactionBlockDocumented} {
		t.Run(name, func(t *testing.T) {
			t.Parallel()
			server := newReplyServer(compactionReply(block, "end_turn"), nil)
			defer server.Close()

			response, err := newTestModel(t, server.URL).Generate(context.Background(), fantasy.Call{
				Prompt:          testPrompt(),
				ProviderOptions: NewProviderOptions(&ProviderOptions{ContextManagement: compactionEdit()}),
			})
			require.NoError(t, err)
			require.Len(t, response.Content, 2)
			reasoning, ok := fantasy.AsContentType[fantasy.ReasoningContent](response.Content[0])
			require.True(t, ok, "the compaction block is not reasoning content")
			require.Equal(t, compactionSummary, reasoning.Text)
			compaction := GetCompactionMetadata(fantasy.ProviderOptions(reasoning.ProviderMetadata))
			require.NotNil(t, compaction)
			require.JSONEq(t, block, string(compaction.Block))
			require.Equal(t, fantasy.FinishReasonStop, response.FinishReason)
		})
	}
}

func TestGenerate_PausedAfterCompaction(t *testing.T) {
	t.Parallel()

	server := newReplyServer(compactionReply(compactionBlockAsSent, "compaction"), nil)
	defer server.Close()

	response, err := newTestModel(t, server.URL).Generate(context.Background(), fantasy.Call{
		Prompt:          testPrompt(),
		ProviderOptions: NewProviderOptions(&ProviderOptions{ContextManagement: compactionEdit()}),
	})
	require.NoError(t, err)
	require.Equal(t, fantasy.FinishReasonStop, response.FinishReason)
	require.Len(t, response.Content, 1)
	_, ok := fantasy.AsContentType[fantasy.ReasoningContent](response.Content[0])
	require.True(t, ok)
}

func TestGenerate_ReportsCompactionIterations(t *testing.T) {
	t.Parallel()

	server := newReplyServer(compactionReply(compactionBlockAsSent, "end_turn"), nil)
	defer server.Close()

	response, err := newTestModel(t, server.URL).Generate(context.Background(), fantasy.Call{
		Prompt:          testPrompt(),
		ProviderOptions: NewProviderOptions(&ProviderOptions{ContextManagement: compactionEdit()}),
	})
	require.NoError(t, err)
	require.Equal(t, &ContextManagementMetadata{
		AppliedEdits: []AppliedContextEdit{},
		Iterations: []UsageIteration{
			{Type: "compaction", InputTokens: 135983, OutputTokens: 1024},
			{Type: "message", InputTokens: 214, OutputTokens: 394},
		},
	}, response.ProviderMetadata[Name])
	require.Equal(t, int64(214), response.Usage.InputTokens, "the response's own usage stays the API's top-level figure")
}

// compactionEvents are the events of a streamed compaction block, as the
// API sent them. With withEncrypted, the deltas also carry encrypted content.
func compactionEvents(withEncrypted bool) []map[string]any {
	half := len(compactionSummary) / 2
	delta := func(content, encrypted string) map[string]any {
		d := map[string]any{"type": "compaction_delta", "content": content}
		if encrypted != "" {
			d["encrypted_content"] = encrypted
		}
		return map[string]any{"type": "content_block_delta", "index": 0, "delta": d}
	}
	enc1, enc2 := "", ""
	if withEncrypted {
		enc1, enc2 = "opaque-", "1"
	}
	return []map[string]any{
		{"type": "content_block_start", "index": 0, "content_block": map[string]any{"type": "compaction", "content": nil}},
		{"type": "ping"},
		delta(compactionSummary[:half], enc1),
		delta(compactionSummary[half:], enc2),
		{"type": "content_block_stop", "index": 0},
	}
}

// compactionStream is a streamed compaction reply as the API sent it.
func compactionStream(withEncrypted bool) []string {
	end := messageDelta("end_turn", map[string]any{"applied_edits": []any{}})
	return replyStream(end, compactionEvents(withEncrypted)...)
}

func TestStream_CompactionBlockBecomesReasoning(t *testing.T) {
	t.Parallel()

	tests := map[string]struct {
		encrypted bool
		want      string
	}{
		"as sent":        {false, compactionBlockAsSent},
		"with encrypted": {true, `{"type":"compaction","content":"` + compactionSummary + `","encrypted_content":"opaque-1"}`},
	}
	for name, tt := range tests {
		t.Run(name, func(t *testing.T) {
			t.Parallel()
			server := newReplyServer(nil, compactionStream(tt.encrypted))
			defer server.Close()

			stream, err := newTestModel(t, server.URL).Stream(context.Background(), fantasy.Call{
				Prompt:          testPrompt(),
				ProviderOptions: NewProviderOptions(&ProviderOptions{ContextManagement: compactionEdit()}),
			})
			require.NoError(t, err)
			var started, ended bool
			var text strings.Builder
			var block json.RawMessage
			deltas := 0
			stream(func(part fantasy.StreamPart) bool {
				switch part.Type {
				case fantasy.StreamPartTypeReasoningStart:
					started = part.ID == "0"
				case fantasy.StreamPartTypeReasoningDelta:
					deltas++
					text.WriteString(part.Delta)
					require.Nil(t, part.ProviderMetadata, "a delta must not replace the block's metadata")
				case fantasy.StreamPartTypeReasoningEnd:
					ended = true
					if m := GetCompactionMetadata(fantasy.ProviderOptions(part.ProviderMetadata)); m != nil {
						block = m.Block
					}
				}
				return true
			})
			require.True(t, started)
			require.True(t, ended)
			require.Equal(t, 2, deltas, "one reasoning delta per compaction_delta, and none for the null start")
			require.Equal(t, compactionSummary, text.String())
			require.JSONEq(t, tt.want, string(block))
		})
	}
}

// compactionPrompt is a conversation whose assistant turn holds a compaction
// block. The block is stored as a ReasoningPart with CompactionMetadata.
func compactionPrompt(block string) fantasy.Prompt {
	return fantasy.Prompt{
		fantasy.NewUserMessage("Here is an inventory."),
		{Role: fantasy.MessageRoleAssistant, Content: []fantasy.MessagePart{
			fantasy.ReasoningPart{Text: compactionSummary, ProviderOptions: fantasy.ProviderOptions{
				Name: &CompactionMetadata{Block: json.RawMessage(block)},
			}},
			fantasy.TextPart{Text: "Item 37 is a green crate from Oslo."},
		}},
		fantasy.NewUserMessage("And item 12?"),
	}
}

func TestGenerate_SendsCompactionBlockBack(t *testing.T) {
	t.Parallel()

	for name, tt := range map[string]struct {
		block         string
		sendReasoning bool
	}{
		"as sent":                 {compactionBlockAsSent, true},
		"documented":              {compactionBlockDocumented, true},
		"with reasoning not sent": {compactionBlockAsSent, false},
	} {
		t.Run(name, func(t *testing.T) {
			t.Parallel()
			server, calls := newCapturingServer(false)
			defer server.Close()

			send := tt.sendReasoning
			_, err := newTestModel(t, server.URL).Generate(context.Background(), fantasy.Call{
				Prompt: compactionPrompt(tt.block),
				ProviderOptions: NewProviderOptions(&ProviderOptions{
					ContextManagement: compactionEdit(),
					SendReasoning:     &send,
				}),
			})
			require.NoError(t, err)

			messages, _ := awaitCaptured(t, calls).body["messages"].([]any)
			require.Len(t, messages, 3)
			assistant := messages[1].(map[string]any)
			content := assistant["content"].([]any)
			require.GreaterOrEqual(t, len(content), 1)
			got, _ := json.Marshal(content[0])
			require.JSONEq(t, tt.block, string(got))
		})
	}
}

func TestGenerate_CompactionBetaFlags(t *testing.T) {
	t.Parallel()

	keep := &ContextAmount{Type: ContextAmountToolUses, Value: 3}
	for name, tt := range map[string]struct {
		cm         *ContextManagement
		want, nope []string
	}{
		"compaction alone": {compactionEdit(), []string{"compact-2026-01-12"}, []string{wantContextManagementBeta}},
		"clearing alone":   {&ContextManagement{Edits: []ContextEdit{{Type: ContextEditClearToolUses, Keep: keep}}}, []string{wantContextManagementBeta}, []string{"compact-2026-01-12"}},
		"both": {&ContextManagement{Edits: []ContextEdit{
			{Type: ContextEditClearToolUses, Keep: keep},
			compactionEdit().Edits[0],
		}}, []string{wantContextManagementBeta, "compact-2026-01-12"}, nil},
	} {
		t.Run(name, func(t *testing.T) {
			t.Parallel()
			server, calls := newCapturingServer(false)
			defer server.Close()

			_, err := newTestModel(t, server.URL).Generate(context.Background(), fantasy.Call{
				Prompt:          testPrompt(),
				ProviderOptions: NewProviderOptions(&ProviderOptions{ContextManagement: tt.cm}),
			})
			require.NoError(t, err)
			betas := awaitCaptured(t, calls).header.Values("Anthropic-Beta")
			for _, b := range tt.want {
				require.Contains(t, betas, b)
			}
			for _, b := range tt.nope {
				require.NotContains(t, betas, b)
			}
		})
	}
}

func TestCompactionMetadata_RoundTrip(t *testing.T) {
	t.Parallel()

	in := fantasy.ProviderMetadata{Name: &CompactionMetadata{Block: json.RawMessage(compactionBlockDocumented)}}
	data, err := json.Marshal(in)
	require.NoError(t, err)
	var raw map[string]json.RawMessage
	require.NoError(t, json.Unmarshal(data, &raw))
	back, err := fantasy.UnmarshalProviderMetadata(raw)
	require.NoError(t, err)
	got, ok := back[Name].(*CompactionMetadata)
	require.True(t, ok)
	require.JSONEq(t, compactionBlockDocumented, string(got.Block))

	_, err = fantasy.UnmarshalProviderMetadata(map[string]json.RawMessage{
		Name: json.RawMessage(`{"type":"anthropic.compaction_metadata","data":"not an object"}`),
	})
	require.Error(t, err)
}

func TestStream_ReportsCompactionIterations(t *testing.T) {
	t.Parallel()

	end := messageDelta("end_turn", map[string]any{"applied_edits": []any{}})
	end["usage"] = map[string]any{"input_tokens": 405, "output_tokens": 633, "iterations": []any{
		map[string]any{"type": "compaction", "input_tokens": 135983, "output_tokens": 338},
		map[string]any{"type": "message", "input_tokens": 405, "output_tokens": 633},
	}}
	chunks := replyStream(end, compactionEvents(false)...)
	server := newReplyServer(nil, chunks)
	defer server.Close()

	require.Equal(t, &ContextManagementMetadata{
		AppliedEdits: []AppliedContextEdit{},
		Iterations: []UsageIteration{
			{Type: "compaction", InputTokens: 135983, OutputTokens: 338},
			{Type: "message", InputTokens: 405, OutputTokens: 633},
		},
	}, finishMetadata(t, server.URL)[Name])
}

func TestOpenCompaction_BuildsTheBlock(t *testing.T) {
	t.Parallel()

	tests := []struct {
		name   string
		start  string
		deltas []string
		want   string
	}{
		{
			name:  "a failed compaction keeps its null content",
			start: `{"type":"content_block_start","index":0,"content_block":{"type":"compaction","content":null}}`,
			want:  `{"type":"compaction","content":null}`,
		},
		{
			name:   "content and encrypted content from the start and the deltas",
			start:  `{"type":"content_block_start","index":0,"content_block":{"type":"compaction","content":"a","encrypted_content":"x","signature":"s"}}`,
			deltas: []string{`{"type":"content_block_delta","index":0,"delta":{"type":"compaction_delta","content":"b","encrypted_content":"y"}}`},
			want:   `{"type":"compaction","content":"ab","encrypted_content":"xy","signature":"s"}`,
		},
		{
			name:   "a one-character summary",
			start:  `{"type":"content_block_start","index":0,"content_block":{"type":"compaction","content":null}}`,
			deltas: []string{`{"type":"content_block_delta","index":0,"delta":{"type":"compaction_delta","content":"z","encrypted_content":"q"}}`},
			want:   `{"type":"compaction","content":"z","encrypted_content":"q"}`,
		},
		{
			name:  "a start without a block still says compaction",
			start: `{"type":"content_block_start","index":0}`,
			want:  `{"type":"compaction"}`,
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			t.Parallel()
			open := newOpenCompaction(tt.start)
			for _, d := range tt.deltas {
				open.delta(d)
			}
			require.JSONEq(t, tt.want, string(open.block()))
		})
	}
}

func TestContextManagementMetadata_IgnoresAMalformedReply(t *testing.T) {
	t.Parallel()

	require.Empty(t, contextManagementMetadata(`{"context_management":{"applied_edits":[]},"usage":"not an object"}`))
	require.Empty(t, contextManagementMetadata(``))
	require.Empty(t, contextManagementMetadata(`{"usage":{"input_tokens":3}}`))
}

// When a caller stops reading partway through a compaction, the stream
// doesn't yield any more parts.
func TestStream_CompactionStopsWhenTheCallerStops(t *testing.T) {
	t.Parallel()

	onDemand := func() []string {
		return replyStream(messageDelta("end_turn", nil),
			onDemandStart(), map[string]any{"type": "content_block_stop", "index": 0})
	}
	for _, tt := range []struct {
		name   string
		chunks []string
		stopAt fantasy.StreamPartType
	}{
		{"threshold start", compactionStream(false), fantasy.StreamPartTypeReasoningStart},
		{"threshold delta", compactionStream(false), fantasy.StreamPartTypeReasoningDelta},
		{"threshold end", compactionStream(false), fantasy.StreamPartTypeReasoningEnd},
		{"on demand delta", onDemand(), fantasy.StreamPartTypeReasoningDelta},
	} {
		stopAt := tt.stopAt
		t.Run(tt.name, func(t *testing.T) {
			t.Parallel()
			server := newReplyServer(nil, tt.chunks)
			defer server.Close()

			stream, err := newTestModel(t, server.URL).Stream(context.Background(), fantasy.Call{
				Prompt:          testPrompt(),
				ProviderOptions: NewProviderOptions(&ProviderOptions{ContextManagement: compactionEdit()}),
			})
			require.NoError(t, err)
			stopped, after := false, 0
			stream(func(part fantasy.StreamPart) bool {
				if stopped {
					after++
					return false
				}
				if part.Type == stopAt {
					stopped = true
					return false
				}
				return true
			})
			require.True(t, stopped, "the stream never sent %s", stopAt)
			require.Zero(t, after, "parts kept coming after the caller stopped")
		})
	}
}

// The stream skips a compaction_delta for a block it never started. It
// doesn't crash.
func TestStream_CompactionDeltaWithoutAStart(t *testing.T) {
	t.Parallel()

	chunks := replyStream(messageDelta("end_turn", nil), map[string]any{
		"type": "content_block_delta", "index": 5,
		"delta": map[string]any{"type": "compaction_delta", "content": "stray"},
	})
	server := newReplyServer(nil, chunks)
	defer server.Close()

	stream, err := newTestModel(t, server.URL).Stream(context.Background(), fantasy.Call{Prompt: testPrompt()})
	require.NoError(t, err)
	var finished bool
	stream(func(part fantasy.StreamPart) bool {
		require.NotEqual(t, fantasy.StreamPartTypeReasoningDelta, part.Type)
		finished = finished || part.Type == fantasy.StreamPartTypeFinish
		return true
	})
	require.True(t, finished)
}

// Compaction on demand. The shapes below are what the API sent for
// claude-haiku-5-5 on 2026-10-21. It sent one signed block with the stop
// reason "compaction". Top-level usage was zero and the cost was in
// iterations. On a stream the whole block came on content_block_start,
// without deltas.

const onDemandSummary = "The user is designing a recipe app's data model: Recipe, Ingredient, Step and RecipeIngredient. They asked for Recipe's field names."

var onDemandBlock = `{"type":"compaction","content":"` + onDemandSummary + `","signature":"EuYBCkQY-sig"}`

// onDemandStart is the content_block_start of onDemandBlock on a stream.
// The whole block arrives on its start.
func onDemandStart() map[string]any {
	return map[string]any{"type": "content_block_start", "index": 0, "content_block": json.RawMessage(onDemandBlock)}
}

func onDemandOptions(instructions *string) fantasy.ProviderOptions {
	return NewProviderOptions(&ProviderOptions{Compaction: &Compaction{Instructions: instructions}})
}

func onDemandReply(content []any, stopReason string) map[string]any {
	reply := mockAnthropicGenerateResponse()
	reply["content"] = content
	reply["stop_reason"] = stopReason
	reply["usage"] = map[string]any{
		"input_tokens": 0, "output_tokens": 0,
		"iterations": []any{map[string]any{"type": "compaction", "input_tokens": 144, "output_tokens": 276}},
	}
	return reply
}

func TestGenerate_SendsCompactionOnDemand(t *testing.T) {
	t.Parallel()

	instructions := "Keep every entity name."
	for name, tt := range map[string]struct {
		instructions *string
		want         map[string]any
	}{
		"default":           {nil, map[string]any{"type": "summarize"}},
		"with instructions": {&instructions, map[string]any{"type": "summarize", "instructions": instructions}},
	} {
		t.Run(name, func(t *testing.T) {
			t.Parallel()
			server, calls := newCapturingServer(false)
			defer server.Close()

			_, err := newTestModel(t, server.URL).Generate(context.Background(), fantasy.Call{
				Prompt:          testPrompt(),
				ProviderOptions: onDemandOptions(tt.instructions),
			})
			require.NoError(t, err)
			call := awaitCaptured(t, calls)
			require.Equal(t, tt.want, call.body["compaction"])
			require.NotContains(t, call.body, "context_management")
			require.Contains(t, call.header.Values("Anthropic-Beta"), "compact-2026-09-04")
			require.Contains(t, call.query, "beta=true")
		})
	}
}

func TestGenerate_NoCompactionOnDemandWithoutIt(t *testing.T) {
	t.Parallel()

	server, calls := newCapturingServer(false)
	defer server.Close()

	_, err := newTestModel(t, server.URL).Generate(context.Background(), fantasy.Call{Prompt: testPrompt()})
	require.NoError(t, err)
	call := awaitCaptured(t, calls)
	require.NotContains(t, call.body, "compaction")
	require.NotContains(t, call.header.Values("Anthropic-Beta"), "compact-2026-09-04")
}

func TestGenerate_CompactionOnDemandRefusesContextManagement(t *testing.T) {
	t.Parallel()

	server, calls := newCapturingServer(false)
	defer server.Close()

	_, err := newTestModel(t, server.URL).Generate(context.Background(), fantasy.Call{
		Prompt: testPrompt(),
		ProviderOptions: NewProviderOptions(&ProviderOptions{
			Compaction:        &Compaction{},
			ContextManagement: bothEdits(),
		}),
	})
	require.Error(t, err)
	require.Contains(t, err.Error(), "compaction not supported with context management")
	select {
	case <-calls:
		t.Fatal("the request was sent anyway")
	default:
	}
}

func TestGenerate_CompactionOnDemandDropsARequiredToolChoice(t *testing.T) {
	t.Parallel()

	for name, tt := range map[string]struct {
		choice   fantasy.ToolChoice
		warning  string
		wantType string
	}{
		"required is dropped":     {fantasy.ToolChoiceRequired, "required tool choice not supported with compaction", ""},
		"a named tool is dropped": {fantasy.ToolChoice(lookupTool.Name), "named tool lookup_weather not supported with compaction", ""},
		"auto is kept":            {fantasy.ToolChoiceAuto, "", "auto"},
	} {
		t.Run(name, func(t *testing.T) {
			t.Parallel()
			server, calls := newCapturingServer(false)
			defer server.Close()

			choice := tt.choice
			response, err := newTestModel(t, server.URL).Generate(context.Background(), fantasy.Call{
				Prompt:          testPrompt(),
				Tools:           []fantasy.Tool{lookupTool},
				ToolChoice:      &choice,
				ProviderOptions: onDemandOptions(nil),
			})
			require.NoError(t, err)
			call := awaitCaptured(t, calls)
			got, _ := call.body["tool_choice"].(map[string]any)
			var warning string
			for _, w := range response.Warnings {
				if w.Setting == "ToolChoice" {
					warning = w.Details
				}
			}
			require.Equal(t, tt.warning, warning)
			if tt.warning != "" {
				require.Empty(t, got)
				return
			}
			require.Equal(t, tt.wantType, got["type"])
		})
	}
}

var lookupTool = fantasy.FunctionTool{
	Name:        "lookup_weather",
	Description: "Look up today's weather for a city.",
	InputSchema: map[string]any{"type": "object", "properties": map[string]any{"city": map[string]any{"type": "string"}}},
}

func TestGenerate_CompactionOnDemandReply(t *testing.T) {
	t.Parallel()

	var block map[string]any
	require.NoError(t, json.Unmarshal([]byte(onDemandBlock), &block))
	server := newReplyServer(onDemandReply([]any{block}, "compaction"), nil)
	defer server.Close()

	response, err := newTestModel(t, server.URL).Generate(context.Background(), fantasy.Call{
		Prompt:          testPrompt(),
		ProviderOptions: onDemandOptions(nil),
	})
	require.NoError(t, err)
	require.Equal(t, fantasy.FinishReasonStop, response.FinishReason)
	require.Len(t, response.Content, 1)
	reasoning, ok := fantasy.AsContentType[fantasy.ReasoningContent](response.Content[0])
	require.True(t, ok)
	require.Equal(t, onDemandSummary, reasoning.Text)
	compaction := GetCompactionMetadata(fantasy.ProviderOptions(reasoning.ProviderMetadata))
	require.NotNil(t, compaction)
	require.JSONEq(t, onDemandBlock, string(compaction.Block))
	require.Equal(t, "compact-2026-09-04", compaction.Beta)
	require.Equal(t, &ContextManagementMetadata{
		Iterations: []UsageIteration{{Type: "compaction", InputTokens: 144, OutputTokens: 276}},
	}, response.ProviderMetadata[Name])
}

func TestGenerate_CompactionOnDemandWithNoSummary(t *testing.T) {
	t.Parallel()

	server := newReplyServer(onDemandReply([]any{}, "max_tokens"), nil)
	defer server.Close()

	response, err := newTestModel(t, server.URL).Generate(context.Background(), fantasy.Call{
		Prompt:          testPrompt(),
		ProviderOptions: onDemandOptions(nil),
	})
	require.NoError(t, err)
	require.Equal(t, fantasy.FinishReasonLength, response.FinishReason)
	require.Empty(t, response.Content)
}

func TestStream_CompactionOnDemandArrivesWhole(t *testing.T) {
	t.Parallel()

	chunks := replyStream(messageDelta("compaction", nil),
		onDemandStart(),
		map[string]any{"type": "ping"},
		map[string]any{"type": "content_block_stop", "index": 0},
	)
	server := newReplyServer(nil, chunks)
	defer server.Close()

	stream, err := newTestModel(t, server.URL).Stream(context.Background(), fantasy.Call{
		Prompt:          testPrompt(),
		ProviderOptions: onDemandOptions(nil),
	})
	require.NoError(t, err)
	var text strings.Builder
	var compaction *CompactionMetadata
	var finish fantasy.FinishReason
	stream(func(part fantasy.StreamPart) bool {
		switch part.Type {
		case fantasy.StreamPartTypeReasoningDelta:
			text.WriteString(part.Delta)
		case fantasy.StreamPartTypeReasoningEnd:
			compaction = GetCompactionMetadata(fantasy.ProviderOptions(part.ProviderMetadata))
		case fantasy.StreamPartTypeFinish:
			finish = part.FinishReason
		}
		return true
	})
	require.Equal(t, onDemandSummary, text.String(), "the summary reaches the reasoning text")
	require.NotNil(t, compaction)
	require.JSONEq(t, onDemandBlock, string(compaction.Block))
	require.Equal(t, "compact-2026-09-04", compaction.Beta)
	require.Equal(t, fantasy.FinishReasonStop, finish)
}

// onDemandHistory is a conversation after on-demand compaction. The block
// comes first as its own assistant message. The next turn follows.
func onDemandHistory(beta string) fantasy.Prompt {
	return fantasy.Prompt{
		{Role: fantasy.MessageRoleAssistant, Content: []fantasy.MessagePart{
			fantasy.ReasoningPart{Text: onDemandSummary, ProviderOptions: fantasy.ProviderOptions{
				Name: &CompactionMetadata{Block: json.RawMessage(onDemandBlock), Beta: beta},
			}},
		}},
		fantasy.NewUserMessage("Now suggest field names for Ingredient."),
	}
}

func TestGenerate_CarriesTheCompactionBlockAndItsBeta(t *testing.T) {
	t.Parallel()

	server, calls := newCapturingServer(false)
	defer server.Close()

	_, err := newTestModel(t, server.URL).Generate(context.Background(), fantasy.Call{Prompt: onDemandHistory("compact-2026-09-04")})
	require.NoError(t, err)
	call := awaitCaptured(t, calls)
	messages := call.body["messages"].([]any)
	first := messages[0].(map[string]any)
	require.Equal(t, "assistant", first["role"])
	got, _ := json.Marshal(first["content"].([]any)[0])
	require.JSONEq(t, onDemandBlock, string(got))
	betas := call.header.Values("Anthropic-Beta")
	require.Contains(t, betas, "compact-2026-09-04")
	require.Contains(t, call.query, "beta=true")
	require.NotContains(t, call.body, "compaction", "carrying a block is not asking for another")
}

func TestGenerate_CarriedBetaIsSentOnce(t *testing.T) {
	t.Parallel()

	server, calls := newCapturingServer(false)
	defer server.Close()

	prompt := onDemandHistory("compact-2026-09-04")
	prompt = append(prompt, onDemandHistory("compact-2026-09-04")...)
	_, err := newTestModel(t, server.URL).Generate(context.Background(), fantasy.Call{
		Prompt:          prompt,
		ProviderOptions: onDemandOptions(nil),
	})
	require.NoError(t, err)
	var n int
	for _, b := range awaitCaptured(t, calls).header.Values("Anthropic-Beta") {
		if b == "compact-2026-09-04" {
			n++
		}
	}
	require.Equal(t, 1, n)
}

func TestGenerate_ABlockWithNoBetaAddsNone(t *testing.T) {
	t.Parallel()

	server, calls := newCapturingServer(false)
	defer server.Close()

	_, err := newTestModel(t, server.URL).Generate(context.Background(), fantasy.Call{Prompt: onDemandHistory("")})
	require.NoError(t, err)
	require.Empty(t, awaitCaptured(t, calls).header.Values("Anthropic-Beta"))
}

func TestCompactionBeta_ByWhatAskedForIt(t *testing.T) {
	t.Parallel()

	keep := &ContextAmount{Type: ContextAmountToolUses, Value: 3}
	require.Equal(t, "", (*ProviderOptions)(nil).compactionBeta())
	require.Equal(t, "", (&ProviderOptions{}).compactionBeta())
	require.Equal(t, "compact-2026-09-04", (&ProviderOptions{Compaction: &Compaction{}}).compactionBeta())
	require.Equal(t, "compact-2026-01-12", (&ProviderOptions{ContextManagement: compactionEdit()}).compactionBeta())
	require.Equal(t, "", (&ProviderOptions{ContextManagement: &ContextManagement{Edits: []ContextEdit{{Type: ContextEditClearToolUses, Keep: keep}}}}).compactionBeta())
}

// When a caller marks the message holding a compaction block for caching,
// the request keeps the breakpoint on that block.
func TestGenerate_CompactionBlockKeepsItsCacheBreakpoint(t *testing.T) {
	t.Parallel()

	for name, cached := range map[string]bool{"marked": true, "not marked": false} {
		t.Run(name, func(t *testing.T) {
			t.Parallel()
			server, calls := newCapturingServer(false)
			defer server.Close()

			prompt := onDemandHistory("compact-2026-09-04")
			if cached {
				prompt[0].ProviderOptions = NewProviderCacheControlOptions(&ProviderCacheControlOptions{CacheControl: CacheControl{Type: "ephemeral"}})
			}
			_, err := newTestModel(t, server.URL).Generate(context.Background(), fantasy.Call{Prompt: prompt})
			require.NoError(t, err)

			first := awaitCaptured(t, calls).body["messages"].([]any)[0].(map[string]any)
			sent := first["content"].([]any)[0].(map[string]any)
			var want map[string]any
			require.NoError(t, json.Unmarshal([]byte(onDemandBlock), &want))
			if cached {
				want["cache_control"] = map[string]any{"type": "ephemeral"}
			}
			require.Equal(t, want, sent, "the block goes back as it came, with only the breakpoint added")
		})
	}
}

func TestCompactionBlockParam_LeavesABlockThatIsNotAnObject(t *testing.T) {
	t.Parallel()

	got, err := json.Marshal(compactionBlockParam(json.RawMessage(`"not an object"`), true))
	require.NoError(t, err)
	require.JSONEq(t, `"not an object"`, string(got))
}

// The API refuses on-demand compaction of a conversation that holds a
// threshold block. In tests on 2026-10-21 the threshold block didn't have a
// signature. So that request is an error before sending.
func TestGenerate_CompactionOnDemandRefusesAThresholdBlock(t *testing.T) {
	t.Parallel()

	for name, tt := range map[string]struct {
		beta    string
		refused bool
	}{
		"a threshold block":  {"compact-2026-01-12", true},
		"an on-demand block": {"compact-2026-09-04", false},
	} {
		t.Run(name, func(t *testing.T) {
			t.Parallel()
			server, calls := newCapturingServer(false)
			defer server.Close()

			_, err := newTestModel(t, server.URL).Generate(context.Background(), fantasy.Call{
				Prompt:          onDemandHistory(tt.beta),
				ProviderOptions: onDemandOptions(nil),
			})
			if !tt.refused {
				require.NoError(t, err)
				awaitCaptured(t, calls)
				return
			}
			require.Error(t, err)
			require.Contains(t, err.Error(), "compaction on demand not supported after threshold compaction")
			select {
			case <-calls:
				t.Fatal("the request was sent anyway")
			default:
			}
		})
	}
}
