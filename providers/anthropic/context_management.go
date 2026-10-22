package anthropic

import (
	"encoding/json"
	"strings"

	"charm.land/fantasy"
)

// ContextEditType names a context editing strategy. The date in each name
// is the strategy's version. The name is sent exactly as written.
type ContextEditType string

const (
	// ContextEditClearToolUses removes the results of old tool calls from
	// what the model sees. It starts when the conversation reaches Trigger.
	// The most recent Keep tool calls keep their results. Its other fields
	// are ClearAtLeast, ExcludeTools and ClearToolInputs.
	ContextEditClearToolUses ContextEditType = "clear_tool_uses_20250919"
	// ContextEditClearThinking removes thinking blocks from older assistant
	// turns in what the model sees. The most recent Keep turns keep theirs.
	// Keep is its only field.
	ContextEditClearThinking ContextEditType = "clear_thinking_20251015"
	// ContextEditCompact has the API summarise the conversation when it
	// reaches Trigger. The reply then starts with a compaction block. Its
	// summary replaces everything before it. The summary is kept as
	// ReasoningContent with the block in CompactionMetadata. The block goes
	// back unchanged on later turns. Callers that store conversations must
	// keep all of that metadata, not just a thinking signature.
	//
	// Its fields are Trigger, Instructions and PauseAfterCompaction. Once a
	// conversation holds its block, every later request must keep this edit.
	// The API refuses the block otherwise. For compaction on demand, see
	// Compaction.
	ContextEditCompact ContextEditType = "compact_20260112"
)

// ContextAmountType is what a ContextAmount counts. Each constant says
// which fields take it.
type ContextAmountType string

const (
	// ContextAmountInputTokens is a number of input tokens. As a Trigger,
	// the edit starts when the prompt reaches that many tokens. As
	// ClearAtLeast, the API clears only when it can remove at least that
	// many tokens.
	ContextAmountInputTokens ContextAmountType = "input_tokens"
	// ContextAmountToolUses is a number of tool uses. As a Trigger, clearing
	// starts when the conversation has that many. As Keep, it's how many of
	// the most recent tool uses keep their results.
	ContextAmountToolUses ContextAmountType = "tool_uses"
	// ContextAmountThinkingTurns is a number of assistant turns. As the Keep
	// of ContextEditClearThinking, it's how many of the most recent turns
	// keep their thinking. Older turns lose it.
	ContextAmountThinkingTurns ContextAmountType = "thinking_turns"
	// ContextAmountAll keeps the thinking of every turn. It's only for the
	// Keep of ContextEditClearThinking. It doesn't take a value. In options
	// parsed from a map, write it as {"type": "all"}. The bare "all" isn't
	// parsed.
	ContextAmountAll ContextAmountType = "all"
)

// ContextManagement configures Anthropic's context editing. It keeps a long
// agent run inside the context window. The request's messages don't change.
// The API applies the edits before the model reads the conversation. The
// clearing edits send the context-management-2025-06-27 beta. Compaction
// sends compact-2026-01-12.
type ContextManagement struct {
	// Edits are the strategies to apply. With several, a
	// ContextEditClearThinking must come first.
	Edits []ContextEdit `json:"edits"`
}

// Context management uses three beta flags. The clearing edits use one.
// Compaction at a threshold uses the second. Compaction on demand uses the
// third.
const (
	contextManagementBeta = "context-management-2025-06-27"
	compactBeta           = "compact-2026-01-12"
	onDemandCompactBeta   = "compact-2026-09-04"
)

// CompactionType names a kind of compaction on demand.
type CompactionType string

// CompactionSummarize summarises the conversation. It's the only
// CompactionType the API takes.
const CompactionSummarize CompactionType = "summarize"

// Compaction asks for a summary of the conversation now, instead of a reply.
// The API summarises every message in the request. It returns one compaction
// block with FinishReasonStop. The summary is kept as ReasoningContent, as
// ContextEditCompact's is. Replace the summarised messages with the
// response's assistant message. The block then comes first in every later
// request. It goes back as it came, with the compact-2026-09-04 beta. When
// the API can't produce a summary, the response comes back empty. Its finish
// reason says why. The conversation then goes on as it was.
//
// Two combinations return an error before sending, because the API refuses
// them. One is Compaction with ContextManagement. The other is Compaction on
// a conversation that holds a threshold compaction block, because that block
// doesn't have a signature.
//
// The API rejects a compaction request whose ToolChoice is
// ToolChoiceRequired or a named tool. Summaries don't run tools. That
// request is sent without a tool choice, with a warning. ToolChoiceAuto and
// ToolChoiceNone are sent as usual.
//
// Send the same system prompt and tools as the rest of the conversation.
// MaxOutputTokens caps the summary and any thinking before it. In tests on
// 11 models (2026-10-21), summaries of a three-message conversation took 229
// to 555 output tokens. With MaxOutputTokens at 16, claude-haiku-5-5
// returned an empty response with FinishReasonLength.
type Compaction struct {
	// Type defaults to CompactionSummarize.
	Type CompactionType `json:"type,omitempty"`
	// Instructions replace the API's own summarisation prompt. They can be
	// up to 16,384 characters.
	Instructions *string `json:"instructions,omitempty"`
}

// wire is the compaction parameter as the request sends it.
func (c *Compaction) wire() map[string]any {
	kind := c.Type
	if kind == "" {
		kind = CompactionSummarize
	}
	wire := map[string]any{"type": kind}
	if c.Instructions != nil {
		wire["instructions"] = *c.Instructions
	}
	return wire
}

// compactionBeta is the beta flag later requests will need if the reply to o
// holds a compaction block. It's "" when o doesn't ask for compaction.
func (o *ProviderOptions) compactionBeta() string {
	if o == nil {
		return ""
	}
	if o.Compaction != nil {
		return onDemandCompactBeta
	}
	if cm := o.contextManagement(); cm != nil {
		for _, edit := range cm.Edits {
			if edit.Type == ContextEditCompact {
				return compactBeta
			}
		}
	}
	return ""
}

// carriedCompactionBetas are the beta flags the compaction blocks in a
// prompt need.
func carriedCompactionBetas(prompt fantasy.Prompt) []string {
	var flags []string
	for _, message := range prompt {
		for _, part := range message.Content {
			if m := GetCompactionMetadata(part.Options()); m != nil && m.Beta != "" {
				flags = appendFlag(flags, m.Beta)
			}
		}
	}
	return flags
}

// appendFlag adds a beta flag unless it is there already.
func appendFlag(flags []string, flag string) []string {
	for _, f := range flags {
		if f == flag {
			return flags
		}
	}
	return append(flags, flag)
}

// betaFlags returns a flag for each kind of edit in cm.Edits. Clearing has
// one and compaction has another.
func (cm *ContextManagement) betaFlags() []string {
	var clearing, compacting bool
	for _, edit := range cm.Edits {
		if edit.Type == ContextEditCompact {
			compacting = true
		} else {
			clearing = true
		}
	}
	var flags []string
	if clearing {
		flags = append(flags, contextManagementBeta)
	}
	if compacting {
		flags = append(flags, compactBeta)
	}
	return flags
}

// ContextEdit is one context editing strategy. Which fields apply depends
// on Type. Fields left unset take the API's defaults.
type ContextEdit struct {
	Type ContextEditType `json:"type"`
	// Trigger is when the edit starts. ContextEditClearToolUses takes input
	// tokens or tool uses. ContextEditCompact takes input tokens only. Its
	// minimum is 50,000 and its default is 150,000.
	Trigger *ContextAmount `json:"trigger,omitempty"`
	// Keep is how much recent content stays. ContextEditClearToolUses counts
	// tool uses. ContextEditClearThinking counts thinking turns, or takes
	// ContextAmountAll.
	Keep *ContextAmount `json:"keep,omitempty"`
	// ClearAtLeast is the minimum number of input tokens a clearing must
	// remove. If the API can't remove that many, it doesn't clear anything.
	// Only ContextEditClearToolUses takes it.
	ClearAtLeast *ContextAmount `json:"clear_at_least,omitempty"`
	// ExcludeTools names tools whose uses are never cleared. Only
	// ContextEditClearToolUses takes it.
	ExcludeTools []string `json:"exclude_tools,omitempty"`
	// ClearToolInputs also removes the inputs of cleared tool calls, for
	// every tool. The API's other form takes a list of tool names. That form
	// isn't supported here. ExtraBody can send it. Only
	// ContextEditClearToolUses takes it.
	ClearToolInputs *bool `json:"clear_tool_inputs,omitempty"`
	// Instructions are added to the API's own instructions for writing the
	// summary. Only ContextEditCompact takes them.
	Instructions *string `json:"instructions,omitempty"`
	// PauseAfterCompaction ends the turn right after the compaction block
	// when true. The finish reason is FinishReasonStop. Send the
	// conversation again to have the model answer. Only ContextEditCompact
	// takes it.
	PauseAfterCompaction *bool `json:"pause_after_compaction,omitempty"`
}

// ContextAmount is a count and its unit. Triggers and limits take it.
type ContextAmount struct {
	Type  ContextAmountType `json:"type"`
	Value int64             `json:"value"`
}

// MarshalJSON leaves out the value for ContextAmountAll. The API takes that
// type without a value.
func (a ContextAmount) MarshalJSON() ([]byte, error) {
	if a.Type == ContextAmountAll {
		return json.Marshal(struct {
			Type ContextAmountType `json:"type"`
		}{a.Type})
	}
	type plain ContextAmount
	return json.Marshal(plain(a))
}

// contextManagement returns the context editing to send, or nil when
// there is none.
func (o *ProviderOptions) contextManagement() *ContextManagement {
	if o == nil || o.ContextManagement == nil || len(o.ContextManagement.Edits) == 0 {
		return nil
	}
	return o.ContextManagement
}

// TypeContextManagementMetadata is the registry type of
// ContextManagementMetadata.
const TypeContextManagementMetadata = Name + ".context_management_metadata"

func init() {
	fantasy.RegisterProviderType(TypeContextManagementMetadata, func(data []byte) (fantasy.ProviderOptionsData, error) {
		var v ContextManagementMetadata
		if err := json.Unmarshal(data, &v); err != nil {
			return nil, err
		}
		return &v, nil
	})
}

// ContextManagementMetadata is what the API cleared or summarised before
// answering. AppliedEdits comes from context_management. Iterations comes
// from usage.iterations, when the API compacted. It's in the response's
// provider metadata whenever the request asked for context management.
// AppliedEdits is empty when nothing needed clearing.
type ContextManagementMetadata struct {
	AppliedEdits []AppliedContextEdit `json:"applied_edits"`
	// Iterations is the usage of each step of a request that compacted. The
	// compaction step comes first, then the message. The response's own usage
	// is the message step's alone. So the compaction's tokens are only here.
	Iterations []UsageIteration `json:"iterations,omitempty"`
}

// UsageIteration is the usage of one step of a request.
type UsageIteration struct {
	// Type is "compaction" or "message".
	Type                     string `json:"type"`
	InputTokens              int64  `json:"input_tokens"`
	OutputTokens             int64  `json:"output_tokens"`
	CacheCreationInputTokens int64  `json:"cache_creation_input_tokens"`
	CacheReadInputTokens     int64  `json:"cache_read_input_tokens"`
}

// AppliedContextEdit is one edit the API applied before the model read the
// conversation.
type AppliedContextEdit struct {
	Type ContextEditType `json:"type"`
	// ClearedInputTokens is how many input tokens the edit removed.
	ClearedInputTokens int64 `json:"cleared_input_tokens"`
	// ClearedToolUses is how many tool uses ContextEditClearToolUses
	// removed.
	ClearedToolUses int64 `json:"cleared_tool_uses,omitempty"`
	// ClearedThinkingTurns is how many turns ContextEditClearThinking
	// removed thinking from.
	ClearedThinkingTurns int64 `json:"cleared_thinking_turns,omitempty"`
}

// Options implements the ProviderOptions interface.
func (*ContextManagementMetadata) Options() {}

// MarshalJSON implements custom JSON marshaling with type info for
// ContextManagementMetadata.
func (m ContextManagementMetadata) MarshalJSON() ([]byte, error) {
	type plain ContextManagementMetadata
	return fantasy.MarshalProviderType(TypeContextManagementMetadata, plain(m))
}

// UnmarshalJSON implements custom JSON unmarshaling with type info for
// ContextManagementMetadata.
func (m *ContextManagementMetadata) UnmarshalJSON(data []byte) error {
	type plain ContextManagementMetadata
	var p plain
	if err := fantasy.UnmarshalProviderType(data, &p); err != nil {
		return err
	}
	*m = ContextManagementMetadata(p)
	return nil
}

// contextManagementMetadata reads context_management and usage.iterations
// from a response's raw JSON. The JSON is a whole message or a stream's
// message_delta. It returns empty metadata when the response has neither.
func contextManagementMetadata(raw string) fantasy.ProviderMetadata {
	var reply struct {
		ContextManagement *struct {
			AppliedEdits []AppliedContextEdit `json:"applied_edits"`
		} `json:"context_management"`
		Usage struct {
			Iterations []UsageIteration `json:"iterations"`
		} `json:"usage"`
	}
	if raw == "" || json.Unmarshal([]byte(raw), &reply) != nil {
		return fantasy.ProviderMetadata{}
	}
	if reply.ContextManagement == nil && len(reply.Usage.Iterations) == 0 {
		return fantasy.ProviderMetadata{}
	}
	metadata := &ContextManagementMetadata{Iterations: reply.Usage.Iterations}
	if reply.ContextManagement != nil {
		metadata.AppliedEdits = reply.ContextManagement.AppliedEdits
	}
	return fantasy.ProviderMetadata{Name: metadata}
}

// TypeCompactionMetadata is the registry type of CompactionMetadata.
const TypeCompactionMetadata = Name + ".compaction_metadata"

func init() {
	fantasy.RegisterProviderType(TypeCompactionMetadata, func(data []byte) (fantasy.ProviderOptionsData, error) {
		var v CompactionMetadata
		if err := json.Unmarshal(data, &v); err != nil {
			return nil, err
		}
		return &v, nil
	})
}

// CompactionMetadata holds a compaction block. Its summary is the text of
// the ReasoningContent it's attached to. Block is the block exactly as the
// API sent it. It goes back unchanged on later turns. Models send different
// fields, so nothing in it is rebuilt.
type CompactionMetadata struct {
	Block json.RawMessage `json:"block"`
	// Beta is the beta flag that later requests carrying the block need. The
	// threshold and on-demand kinds use different flags.
	Beta string `json:"beta,omitempty"`
}

// Options implements the ProviderOptions interface.
func (*CompactionMetadata) Options() {}

// MarshalJSON implements custom JSON marshaling with type info for
// CompactionMetadata.
func (m CompactionMetadata) MarshalJSON() ([]byte, error) {
	type plain CompactionMetadata
	return fantasy.MarshalProviderType(TypeCompactionMetadata, plain(m))
}

// UnmarshalJSON implements custom JSON unmarshaling with type info for
// CompactionMetadata.
func (m *CompactionMetadata) UnmarshalJSON(data []byte) error {
	type plain CompactionMetadata
	var p plain
	if err := fantasy.UnmarshalProviderType(data, &p); err != nil {
		return err
	}
	*m = CompactionMetadata(p)
	return nil
}

// GetCompactionMetadata returns the compaction block a message part
// carries, or nil.
func GetCompactionMetadata(options fantasy.ProviderOptions) *CompactionMetadata {
	if m, ok := options[Name].(*CompactionMetadata); ok {
		return m
	}
	return nil
}

// compactionContent turns a compaction block from a whole reply into
// ReasoningContent. The summary is the text. The block and its beta flag
// are the metadata.
func compactionContent(raw, beta string) fantasy.ReasoningContent {
	var block struct {
		Content *string `json:"content"`
	}
	_ = json.Unmarshal([]byte(raw), &block)
	text := ""
	if block.Content != nil {
		text = *block.Content
	}
	return fantasy.ReasoningContent{
		Text:             text,
		ProviderMetadata: fantasy.ProviderMetadata{Name: &CompactionMetadata{Block: json.RawMessage(raw), Beta: beta}},
	}
}

// openCompaction is a compaction block that a stream has started and not
// ended. It holds the block from content_block_start and what
// compaction_delta has added. When compaction fails, the start's content is
// null. So the block gets content only when the start or a delta carried
// content.
type openCompaction struct {
	start                    map[string]any
	content, encrypted       strings.Builder
	hasContent, hasEncrypted bool
}

// newOpenCompaction reads the block from a content_block_start event.
func newOpenCompaction(event string) *openCompaction {
	var start struct {
		ContentBlock map[string]any `json:"content_block"`
	}
	_ = json.Unmarshal([]byte(event), &start)
	open := &openCompaction{start: start.ContentBlock}
	if content, ok := open.start["content"].(string); ok {
		open.content.WriteString(content)
		open.hasContent = true
	}
	if encrypted, ok := open.start["encrypted_content"].(string); ok {
		open.encrypted.WriteString(encrypted)
		open.hasEncrypted = true
	}
	return open
}

// delta adds a compaction_delta event and returns the summary text it
// carried.
func (o *openCompaction) delta(event string) string {
	var e struct {
		Delta struct {
			Content          *string `json:"content"`
			EncryptedContent *string `json:"encrypted_content"`
		} `json:"delta"`
	}
	_ = json.Unmarshal([]byte(event), &e)
	text := ""
	if e.Delta.Content != nil {
		text = *e.Delta.Content
		o.content.WriteString(text)
		o.hasContent = true
	}
	if e.Delta.EncryptedContent != nil {
		o.encrypted.WriteString(*e.Delta.EncryptedContent)
		o.hasEncrypted = true
	}
	return text
}

// block is the compaction block the stream built. It has the start block's
// fields, plus the summary and encrypted content from the deltas.
func (o *openCompaction) block() json.RawMessage {
	block := map[string]any{}
	for k, v := range o.start {
		block[k] = v
	}
	block["type"] = "compaction"
	if o.hasContent {
		block["content"] = o.content.String()
	}
	if o.hasEncrypted {
		block["encrypted_content"] = o.encrypted.String()
	}
	data, _ := json.Marshal(block)
	return data
}
