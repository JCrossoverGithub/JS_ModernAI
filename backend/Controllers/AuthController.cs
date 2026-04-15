using System.IdentityModel.Tokens.Jwt;
using System.Security.Claims;
using System.Text;
using Backend.Models;
using Microsoft.AspNetCore.Identity;
using Microsoft.AspNetCore.Mvc;
using Microsoft.IdentityModel.Tokens;

namespace Backend.Controllers;

/// <summary>
/// Authentication controller handling user registration and login.
/// Issues JWT Bearer tokens on successful authentication.
/// These endpoints do NOT require an existing token.
/// </summary>
[ApiController]
[Route("api/[controller]")]
public class AuthController : ControllerBase
{
    private readonly UserManager<IdentityUser> _userManager;
    private readonly IConfiguration _config;

    public AuthController(UserManager<IdentityUser> userManager, IConfiguration config)
    {
        _userManager = userManager;
        _config = config;
    }

    [HttpPost("register")]
    public async Task<IActionResult> Register([FromBody] RegisterRequest req)
    {
        var user = new IdentityUser { UserName = req.Username, Email = req.Email };
        var result = await _userManager.CreateAsync(user, req.Password);

        if (!result.Succeeded)
            return BadRequest(new { errors = result.Errors.Select(e => e.Description) });

        var token = GenerateJwt(user);
        return Ok(new AuthResponse(token, user.UserName!, user.Id));
    }

    [HttpPost("login")]
    public async Task<IActionResult> Login([FromBody] LoginRequest req)
    {
        var user = await _userManager.FindByEmailAsync(req.Email);
        if (user == null || !await _userManager.CheckPasswordAsync(user, req.Password))
            return Unauthorized(new { error = "Invalid email or password." });

        var token = GenerateJwt(user);
        return Ok(new AuthResponse(token, user.UserName!, user.Id));
    }

    private string GenerateJwt(IdentityUser user)
    {
        var key = new SymmetricSecurityKey(Encoding.UTF8.GetBytes(_config["Jwt:Key"]!));
        var creds = new SigningCredentials(key, SecurityAlgorithms.HmacSha256);
        var expires = DateTime.UtcNow.AddMinutes(double.Parse(_config["Jwt:ExpirationMinutes"]!));

        var claims = new[]
        {
            new Claim(JwtRegisteredClaimNames.Sub, user.Id),
            new Claim(JwtRegisteredClaimNames.Email, user.Email!),
            new Claim(ClaimTypes.NameIdentifier, user.Id),
            new Claim(ClaimTypes.Name, user.UserName!),
        };

        var token = new JwtSecurityToken(
            issuer: _config["Jwt:Issuer"],
            audience: _config["Jwt:Audience"],
            claims: claims,
            expires: expires,
            signingCredentials: creds
        );

        return new JwtSecurityTokenHandler().WriteToken(token);
    }
}
