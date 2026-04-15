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
    public async Task<IActionResult> Upload(IFormFile file)
    {
        if (file == null || file.Length == 0)
            return BadRequest(new { error = "No file provided." });

        using var stream = file.OpenReadStream();
        var result = await _ai.UploadDocumentAsync(stream, file.FileName);
        return Content(result, "application/json");
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
