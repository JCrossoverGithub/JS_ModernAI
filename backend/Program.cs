// =============================================================================
// Program.cs — ASP.NET Core 8 application entry point for the LibraryAI backend.
//
// This file configures:
//   - SQLite database via EF Core (auto-created on first run)
//   - ASP.NET Core Identity for user management
//   - JWT Bearer authentication (with SignalR query-string support)
//   - CORS policy for the React frontend
//   - Typed HttpClient for the Python AI microservice
//   - SignalR hub for real-time chat streaming
//   - REST controllers for auth, memory, and documents
// =============================================================================

using System.Security.Claims;
using System.Text;
using Backend.Data;
using Backend.Hubs;
using Backend.Services;
using Microsoft.AspNetCore.Authentication.JwtBearer;
using Microsoft.AspNetCore.Identity;
using Microsoft.EntityFrameworkCore;
using Microsoft.IdentityModel.Tokens;

var builder = WebApplication.CreateBuilder(args);

// --- Database ---
builder.Services.AddDbContext<AppDbContext>(options =>
    options.UseSqlite(builder.Configuration.GetConnectionString("DefaultConnection")));

// --- Identity ---
builder.Services.AddIdentity<IdentityUser, IdentityRole>(options =>
{
    options.Password.RequireDigit = false;
    options.Password.RequiredLength = 6;
    options.Password.RequireNonAlphanumeric = false;
    options.Password.RequireUppercase = false;
})
.AddEntityFrameworkStores<AppDbContext>()
.AddDefaultTokenProviders();

// --- JWT Auth ---
var jwtKey = builder.Configuration["Jwt:Key"]!;
builder.Services.AddAuthentication(options =>
{
    options.DefaultAuthenticateScheme = JwtBearerDefaults.AuthenticationScheme;
    options.DefaultChallengeScheme = JwtBearerDefaults.AuthenticationScheme;
})
.AddJwtBearer(options =>
{
    options.TokenValidationParameters = new TokenValidationParameters
    {
        ValidateIssuer = true,
        ValidateAudience = true,
        ValidateLifetime = true,
        ValidateIssuerSigningKey = true,
        ValidIssuer = builder.Configuration["Jwt:Issuer"],
        ValidAudience = builder.Configuration["Jwt:Audience"],
        IssuerSigningKey = new SymmetricSecurityKey(Encoding.UTF8.GetBytes(jwtKey)),
        NameClaimType = ClaimTypes.NameIdentifier,
    };

    // Allow SignalR to receive the JWT via query string
    options.Events = new JwtBearerEvents
    {
        OnMessageReceived = context =>
        {
            var accessToken = context.Request.Query["access_token"];
            var path = context.HttpContext.Request.Path;
            if (!string.IsNullOrEmpty(accessToken) && path.StartsWithSegments("/hubs/chat"))
            {
                context.Token = accessToken;
            }
            return Task.CompletedTask;
        }
    };
});

// --- CORS ---
var allowedOrigins = builder.Configuration["AllowedOrigins"]?.Split(",") ?? ["http://localhost:5173"];
builder.Services.AddCors(options =>
{
    options.AddDefaultPolicy(policy =>
    {
        policy.WithOrigins(allowedOrigins)
              .AllowAnyHeader()
              .AllowAnyMethod()
              .AllowCredentials();
    });
});

// --- Python AI Service ---
builder.Services.AddHttpClient<PythonAIService>(client =>
{
    client.BaseAddress = new Uri(builder.Configuration["PythonAI:BaseUrl"] ?? "http://localhost:8100");
    client.Timeout = TimeSpan.FromMinutes(5); // LLM responses can be slow
});

// --- SignalR + Controllers ---
builder.Services.AddSignalR();
builder.Services.AddControllers();

// --- File upload limits ---
builder.WebHost.ConfigureKestrel(options =>
{
    options.Limits.MaxRequestBodySize = 200 * 1024 * 1024; // 200 MB
});
builder.Services.Configure<Microsoft.AspNetCore.Http.Features.FormOptions>(options =>
{
    options.MultipartBodyLengthLimit = 200 * 1024 * 1024; // 200 MB
});

var app = builder.Build();

// --- Auto-create database ---
using (var scope = app.Services.CreateScope())
{
    var db = scope.ServiceProvider.GetRequiredService<AppDbContext>();
    db.Database.EnsureCreated();
}

app.UseCors();
app.UseAuthentication();
app.UseAuthorization();

app.MapControllers();
app.MapHub<ChatHub>("/hubs/chat");

app.Run();
