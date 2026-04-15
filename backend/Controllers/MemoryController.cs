using System.Security.Claims;
using Backend.Models;
using Backend.Services;
using Microsoft.AspNetCore.Authorization;
using Microsoft.AspNetCore.Mvc;

namespace Backend.Controllers;

/// <summary>
/// Memory management controller. All operations are scoped to the authenticated
/// user's ID (extracted from JWT claims). Delegates to the Python AI service.
/// </summary>
[ApiController]
[Route("api/[controller]")]
[Authorize]
public class MemoryController : ControllerBase
{
    private readonly PythonAIService _ai;

    public MemoryController(PythonAIService ai)
    {
        _ai = ai;
    }

    private string UserId => User.FindFirstValue(ClaimTypes.NameIdentifier)!;

    [HttpPost("facts")]
    public async Task<IActionResult> SaveFact([FromBody] FactRequest req)
    {
        var result = await _ai.SaveFactAsync(UserId, req.Fact);
        return Ok(result);
    }

    [HttpGet("facts")]
    public async Task<IActionResult> ListFacts()
    {
        var result = await _ai.ListFactsAsync(UserId);
        return Content(result, "application/json");
    }

    [HttpDelete("facts")]
    public async Task<IActionResult> DeleteFact([FromQuery] string keyword)
    {
        var result = await _ai.DeleteFactAsync(UserId, keyword);
        return Content(result, "application/json");
    }

    [HttpPost("wipe")]
    public async Task<IActionResult> Wipe()
    {
        await _ai.WipeMemoryAsync(UserId);
        return Ok(new { message = "Memory wiped." });
    }

    [HttpPost("clear-buffer")]
    public async Task<IActionResult> ClearBuffer()
    {
        await _ai.ClearBufferAsync(UserId);
        return Ok(new { message = "Buffer cleared." });
    }
}
