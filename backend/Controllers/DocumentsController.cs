using Backend.Services;
using Microsoft.AspNetCore.Authorization;
using Microsoft.AspNetCore.Mvc;

namespace Backend.Controllers;

/// <summary>
/// Document library controller for uploading, listing, and removing documents.
/// Documents are forwarded to the Python AI service for chunking and embedding.
/// </summary>
[ApiController]
[Route("api/[controller]")]
[Authorize]
public class DocumentsController : ControllerBase
{
    private readonly PythonAIService _ai;

    public DocumentsController(PythonAIService ai)
    {
        _ai = ai;
    }

    [HttpPost("upload")]
    [DisableRequestSizeLimit]
    public async Task Upload(IFormFile file)
    {
        if (file == null || file.Length == 0)
        {
            Response.StatusCode = 400;
            await Response.WriteAsync("{\"error\":\"No file provided.\"}");
            return;
        }

        Response.ContentType = "text/event-stream";
        Response.Headers["Cache-Control"] = "no-cache";
        Response.Headers["Connection"] = "keep-alive";

        using var stream = file.OpenReadStream();
        await _ai.UploadDocumentAsync(stream, file.FileName, async (json) =>
        {
            await Response.WriteAsync($"data: {json}\n\n");
            await Response.Body.FlushAsync();
        });
    }

    [HttpGet]
    public async Task<IActionResult> List()
    {
        var result = await _ai.ListDocumentsAsync();
        return Content(result, "application/json");
    }

    [HttpDelete("{filename}")]
    public async Task<IActionResult> Remove(string filename)
    {
        var result = await _ai.RemoveDocumentAsync(filename);
        return Content(result, "application/json");
    }
}
