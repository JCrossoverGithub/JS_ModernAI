namespace Backend.Models;

/// <summary>Registration request submitted from the frontend.</summary>
/// <param name="Username">Display name for the user.</param>
/// <param name="Email">Email address (used for login).</param>
/// <param name="Password">Password (min 6 characters).</param>
public record RegisterRequest(string Username, string Email, string Password);

/// <summary>Login request submitted from the frontend.</summary>
/// <param name="Email">Email address.</param>
/// <param name="Password">Password.</param>
public record LoginRequest(string Email, string Password);

/// <summary>Successful authentication response containing a JWT.</summary>
/// <param name="Token">JWT Bearer token string.</param>
/// <param name="Username">The user's display name.</param>
/// <param name="UserId">The user's unique Identity ID (GUID).</param>
public record AuthResponse(string Token, string Username, string UserId);
