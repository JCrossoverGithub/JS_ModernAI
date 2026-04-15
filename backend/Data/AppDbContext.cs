using Microsoft.AspNetCore.Identity.EntityFrameworkCore;
using Microsoft.EntityFrameworkCore;

namespace Backend.Data;

/// <summary>
/// Entity Framework Core database context for ASP.NET Core Identity.
/// Uses SQLite as the backing store. The database is auto-created on startup
/// via <c>EnsureCreated()</c> in Program.cs.
/// </summary>
public class AppDbContext : IdentityDbContext
{
    public AppDbContext(DbContextOptions<AppDbContext> options) : base(options) { }
}
