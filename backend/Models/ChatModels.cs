namespace Backend.Models;

/// <summary>Chat query sent from the frontend via REST (not used for SignalR).</summary>
/// <param name="Query">The user's question.</param>
/// <param name="Mode">Search mode: default, strict, chat, or web.</param>
public record ChatRequest(string Query, string Mode = "default");

/// <summary>Request to save a personal fact.</summary>
/// <param name="Fact">The fact text to remember.</param>
public record FactRequest(string Fact);

/// <summary>Request to delete facts by keyword.</summary>
/// <param name="Keyword">Case-insensitive keyword to match against stored facts.</param>
public record KeywordRequest(string Keyword);
