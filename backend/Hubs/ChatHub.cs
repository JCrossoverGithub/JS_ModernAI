using System.Text.Json;
using Backend.Services;
using Microsoft.AspNetCore.Authorization;
using Microsoft.AspNetCore.SignalR;

namespace Backend.Hubs;

/// <summary>
/// SignalR hub for real-time chat streaming. Bridges the React frontend (WebSocket)
/// to the Python AI service (SSE). Requires JWT authentication.
/// 
/// Client → Server: <c>SendMessage(query, mode)</c>
/// Server → Client: <c>ReceiveEvent(jsonString)</c> for each SSE event
/// </summary>
[Authorize]
public class ChatHub : Hub
{
    private readonly PythonAIService _ai;

    public ChatHub(PythonAIService ai)
    {
        _ai = ai;
    }

    /// <summary>
    /// Process a chat query by streaming it through the Python AI service.
    /// Each SSE event from Python is forwarded to the caller via ReceiveEvent.
    /// </summary>
    /// <param name="query">The user's question.</param>
    /// <param name="mode">Search mode: default, strict, chat, or web.</param>
    public async Task SendMessage(string query, string mode)
    {
        var userId = Context.UserIdentifier ?? Context.ConnectionId;

        var eventStream = mode == "research"
            ? _ai.StreamResearchAsync(userId, query)
            : _ai.StreamChatAsync(userId, query, mode);

        await foreach (var jsonEvent in eventStream)
        {
            await Clients.Caller.SendAsync("ReceiveEvent", jsonEvent);
        }
    }
}
