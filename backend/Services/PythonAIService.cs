using System.Net.Http.Headers;
using System.Text;
using System.Text.Json;

namespace Backend.Services;

/// <summary>
/// Typed HttpClient service that communicates with the Python FastAPI AI service.
/// Registered via DI with a base URL configured in appsettings.json (PythonAI:BaseUrl).
/// All AI logic is delegated to Python — this service only handles HTTP transport.
/// </summary>
public class PythonAIService
{
    private readonly HttpClient _http;
    private readonly JsonSerializerOptions _jsonOpts = new() { PropertyNameCaseInsensitive = true };

    public PythonAIService(HttpClient http)
    {
        _http = http;
    }

    // --- Streaming chat via SSE ---
    /// <summary>
    /// Stream chat events from the Python AI service as an async enumerable of JSON strings.
    /// Each yielded string is one SSE "data:" payload (a JSON object with a "type" field).
    /// </summary>
    public async IAsyncEnumerable<string> StreamChatAsync(string userId, string query, string mode, string? folderContext = null)
    {
        var payload = JsonSerializer.Serialize(new { query, mode, user_id = userId, folder_context = folderContext ?? "" });
        var request = new HttpRequestMessage(HttpMethod.Post, "/chat/stream")
        {
            Content = new StringContent(payload, Encoding.UTF8, "application/json")
        };

        using var response = await _http.SendAsync(request, HttpCompletionOption.ResponseHeadersRead);
        response.EnsureSuccessStatusCode();

        using var stream = await response.Content.ReadAsStreamAsync();
        using var reader = new StreamReader(stream);

        while (!reader.EndOfStream)
        {
            var line = await reader.ReadLineAsync();
            if (line == null) break;
            if (line.StartsWith("data: "))
            {
                yield return line["data: ".Length..];
            }
        }
    }

    /// <summary>
    /// Stream academic research events from the Python AI service as an async enumerable of JSON strings.
    /// Routes to the /research/stream endpoint which searches Semantic Scholar and arXiv.
    /// </summary>
    public async IAsyncEnumerable<string> StreamResearchAsync(string userId, string query, string[]? sources = null, string? folderContext = null)
    {
        var payload = JsonSerializer.Serialize(new { query, user_id = userId, sources, folder_context = folderContext ?? "" });
        var request = new HttpRequestMessage(HttpMethod.Post, "/research/stream")
        {
            Content = new StringContent(payload, Encoding.UTF8, "application/json")
        };

        using var response = await _http.SendAsync(request, HttpCompletionOption.ResponseHeadersRead);
        response.EnsureSuccessStatusCode();

        using var stream = await response.Content.ReadAsStreamAsync();
        using var reader = new StreamReader(stream);

        while (!reader.EndOfStream)
        {
            var line = await reader.ReadLineAsync();
            if (line == null) break;
            if (line.StartsWith("data: "))
            {
                yield return line["data: ".Length..];
            }
        }
    }

    // --- Memory ---
    public async Task<string> SaveFactAsync(string userId, string fact)
    {
        var payload = JsonSerializer.Serialize(new { fact, user_id = userId });
        var resp = await _http.PostAsync("/memory/save", new StringContent(payload, Encoding.UTF8, "application/json"));
        resp.EnsureSuccessStatusCode();
        return await resp.Content.ReadAsStringAsync();
    }

    public async Task<string> ListFactsAsync(string userId)
    {
        var resp = await _http.GetAsync($"/memory/facts/{userId}");
        resp.EnsureSuccessStatusCode();
        return await resp.Content.ReadAsStringAsync();
    }

    public async Task<string> DeleteFactAsync(string userId, string keyword)
    {
        var resp = await _http.DeleteAsync($"/memory/facts/{userId}?keyword={Uri.EscapeDataString(keyword)}");
        resp.EnsureSuccessStatusCode();
        return await resp.Content.ReadAsStringAsync();
    }

    public async Task WipeMemoryAsync(string userId)
    {
        var payload = JsonSerializer.Serialize(new { user_id = userId });
        var resp = await _http.PostAsync("/memory/wipe", new StringContent(payload, Encoding.UTF8, "application/json"));
        resp.EnsureSuccessStatusCode();
    }

    public async Task ClearBufferAsync(string userId)
    {
        var payload = JsonSerializer.Serialize(new { user_id = userId });
        var resp = await _http.PostAsync("/memory/clear-buffer", new StringContent(payload, Encoding.UTF8, "application/json"));
        resp.EnsureSuccessStatusCode();
    }

    // --- Documents ---
    /// <summary>
    /// Upload a document to the Python AI service. The Python endpoint now returns SSE events
    /// for progress tracking. This method reads the stream, forwarding progress events via the
    /// optional callback, and returns the final result JSON.
    /// </summary>
    public async Task<string> UploadDocumentAsync(Stream fileStream, string fileName, string userId, Func<string, Task>? onProgress = null)
    {
        using var content = new MultipartFormDataContent();
        var fileContent = new StreamContent(fileStream);
        fileContent.Headers.ContentType = new MediaTypeHeaderValue("application/octet-stream");
        content.Add(fileContent, "file", fileName);
        content.Add(new StringContent(userId), "user_id");

        using var response = await _http.SendAsync(
            new HttpRequestMessage(HttpMethod.Post, "/documents/upload") { Content = content },
            HttpCompletionOption.ResponseHeadersRead);
        response.EnsureSuccessStatusCode();

        using var stream = await response.Content.ReadAsStreamAsync();
        using var reader = new StreamReader(stream);

        string lastEvent = "{}";
        while (!reader.EndOfStream)
        {
            var line = await reader.ReadLineAsync();
            if (line == null) break;
            if (!line.StartsWith("data: ")) continue;

            var json = line["data: ".Length..];
            lastEvent = json;
            if (onProgress != null) await onProgress(json);
        }

        return lastEvent;
    }

    public async Task<string> ListDocumentsAsync()
    {
        var resp = await _http.GetAsync("/documents");
        resp.EnsureSuccessStatusCode();
        return await resp.Content.ReadAsStringAsync();
    }

    public async Task<string> RemoveDocumentAsync(string filename)
    {
        var resp = await _http.DeleteAsync($"/documents/{Uri.EscapeDataString(filename)}");
        resp.EnsureSuccessStatusCode();
        return await resp.Content.ReadAsStringAsync();
    }
}
