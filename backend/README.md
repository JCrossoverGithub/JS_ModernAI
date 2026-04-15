# Backend — ASP.NET Core 8 Web API

The backend is the API gateway and authentication layer. It sits between the React frontend and the Python AI service, handling user registration/login (JWT), proxying AI requests via SignalR for real-time streaming, and managing document/memory CRUD operations.

## Table of Contents

- [Overview](#overview)
- [Architecture](#architecture)
- [Configuration](#configuration)
- [Authentication](#authentication)
- [API Reference](#api-reference)
- [SignalR Hub](#signalr-hub)
- [Services](#services)
- [Database](#database)
- [Running Locally](#running-locally)
- [Docker](#docker)
- [Troubleshooting](#troubleshooting)

---

## Overview

| Component       | Technology                                        |
| --------------- | ------------------------------------------------- |
| Framework       | ASP.NET Core 8 (Minimal Hosting)                  |
| Authentication  | ASP.NET Core Identity + JWT Bearer tokens          |
| Real-time       | SignalR WebSocket hub                              |
| Database        | SQLite via Entity Framework Core                   |
| HTTP Client     | `HttpClient` (typed) to Python AI service          |

The backend does **not** contain any AI logic. It delegates all intelligence to the Python FastAPI service and acts purely as an authentication/authorization gateway with real-time streaming capabilities.

---

## Architecture

```
  React Frontend
       │
       ├── REST (fetch) ──────▶  Controllers
       │                         ├─ AuthController    (register/login)
       │                         ├─ MemoryController  (facts CRUD)
       │                         └─ DocumentsController (upload/list/delete)
       │
       └── WebSocket ──────────▶  ChatHub (SignalR)
                                     │
                                     ▼
                              PythonAIService
                              (HttpClient → FastAPI)
                                     │
                                     ▼
                              Python AI Service
                              (port 8100)
```

**Request flow for chat:**
1. React opens a SignalR WebSocket to `/hubs/chat` (JWT in query string)
2. React calls `hub.invoke("SendMessage", query, mode)`
3. `ChatHub` extracts the user ID from the JWT and calls `PythonAIService.StreamChatAsync()`
4. `PythonAIService` POSTs to Python's `/chat/stream` and reads the SSE response line-by-line
5. Each SSE event is forwarded to the React client via `Clients.Caller.SendAsync("ReceiveEvent", json)`

**Request flow for REST endpoints:**
1. React sends an HTTP request with `Authorization: Bearer <token>` header
2. Controller extracts `UserId` from JWT claims
3. Controller calls `PythonAIService` method which forwards to the Python service
4. Response JSON is returned directly to the client

---

## Configuration

Configuration is stored in `appsettings.json` and can be overridden with environment variables (using `__` as section separator):

```json
{
  "ConnectionStrings": {
    "DefaultConnection": "Data Source=app.db"
  },
  "Jwt": {
    "Key": "CHANGE_THIS_TO_A_RANDOM_64_CHAR_SECRET_KEY_IN_PRODUCTION_1234567890",
    "Issuer": "LibraryAI",
    "Audience": "LibraryAI-Client",
    "ExpirationMinutes": 1440
  },
  "PythonAI": {
    "BaseUrl": "http://localhost:8100"
  },
  "AllowedOrigins": "http://localhost:5173"
}
```

| Key                                      | Default                        | Description                                      |
| ---------------------------------------- | ------------------------------ | ------------------------------------------------ |
| `ConnectionStrings:DefaultConnection`    | `Data Source=app.db`           | SQLite database file path                        |
| `Jwt:Key`                                | *(placeholder)*                | HMAC-SHA256 signing key. **Change in production** |
| `Jwt:Issuer`                             | `LibraryAI`                   | Token issuer claim                               |
| `Jwt:Audience`                           | `LibraryAI-Client`            | Token audience claim                             |
| `Jwt:ExpirationMinutes`                  | `1440` (24 hours)             | Token lifetime                                   |
| `PythonAI:BaseUrl`                       | `http://localhost:8100`        | URL of the Python FastAPI service                |
| `AllowedOrigins`                         | `http://localhost:5173`        | Comma-separated CORS origins                     |

**Environment variable override example:**
```bash
Jwt__Key=my-production-secret-key-that-is-at-least-32-characters-long
PythonAI__BaseUrl=http://ai-service:8100
```

---

## Authentication

The backend uses **ASP.NET Core Identity** for user management with **JWT Bearer tokens** for stateless authentication.

### Password Policy

Relaxed for development. Configured in `Program.cs`:
- Minimum length: 6 characters
- No digit required
- No uppercase required
- No special character required

### Token Format

Tokens are signed with HMAC-SHA256 and contain these claims:

| Claim                         | Description                        |
| ----------------------------- | ---------------------------------- |
| `sub` (Subject)               | The user's Identity ID (GUID)      |
| `email`                       | The user's email address           |
| `http://...nameidentifier`    | Duplicate of `sub` for .NET compat |
| `http://...name`              | The user's username                |

### SignalR Authentication

SignalR WebSocket connections cannot send HTTP headers during the handshake. The JWT is passed as a query parameter:

```
wss://localhost:5000/hubs/chat?access_token=eyJhbG...
```

The `OnMessageReceived` event handler in `Program.cs` extracts the token from the query string for requests to `/hubs/chat`.

---

## API Reference

### Auth Endpoints (No token required)

#### `POST /api/auth/register`

Create a new user account.

**Request:**
```json
{
  "username": "john",
  "email": "john@example.com",
  "password": "secret123"
}
```

**Success (200):**
```json
{
  "token": "eyJhbGciOiJIUzI1NiIs...",
  "username": "john",
  "userId": "a1b2c3d4-e5f6-7890-abcd-ef1234567890"
}
```

**Failure (400):**
```json
{
  "errors": ["Passwords must be at least 6 characters.", "Email 'john@example.com' is already taken."]
}
```

#### `POST /api/auth/login`

Authenticate and receive a JWT.

**Request:**
```json
{
  "email": "john@example.com",
  "password": "secret123"
}
```

**Success (200):** Same shape as register.

**Failure (401):**
```json
{ "error": "Invalid email or password." }
```

---

### Memory Endpoints (JWT required)

All memory endpoints require `Authorization: Bearer <token>`. The user ID is extracted from the token — users can only access their own data.

#### `POST /api/memory/facts`

Save a personal fact.

```json
{ "fact": "My favorite language is C#" }
```

#### `GET /api/memory/facts`

List all saved facts for the authenticated user.

```json
{ "facts": ["My favorite language is C#", "I work at Contoso"] }
```

#### `DELETE /api/memory/facts?keyword=favorite`

Delete facts containing the keyword.

```json
{ "deleted": 1 }
```

#### `POST /api/memory/wipe`

Irreversibly delete all memory (facts + conversation history) for the authenticated user. No request body needed.

#### `POST /api/memory/clear-buffer`

Clear only the in-memory short-term chat buffer. No request body needed.

---

### Document Endpoints (JWT required)

#### `POST /api/documents/upload`

Upload a document to the AI library.

- **Content-Type:** `multipart/form-data`
- **Field:** `file` — `.pdf`, `.txt`, `.docx`, or `.doc`

**Success (200):**
```json
{ "success": true, "chunks": 42, "filename": "paper.pdf" }
```

#### `GET /api/documents`

List all documents in the library.

```json
{ "documents": ["paper.pdf", "notes.txt"] }
```

#### `DELETE /api/documents/{filename}`

Remove a document and all its vector embeddings.

---

## SignalR Hub

### Connection

```
URL: /hubs/chat?access_token=<jwt>
Transport: WebSocket (preferred), Server-Sent Events, Long Polling
```

### Server Methods (client → server)

#### `SendMessage(query: string, mode: string)`

Send a chat query. The server will stream back events via `ReceiveEvent`.

| Parameter | Type   | Description                              |
| --------- | ------ | ---------------------------------------- |
| `query`   | string | The user's question                      |
| `mode`    | string | `default`, `strict`, `chat`, or `web`    |

### Client Methods (server → client)

#### `ReceiveEvent(jsonString: string)`

Receives a JSON string representing one event from the AI pipeline. Parse it to get the event type:

```json
{"type": "token", "content": "Hello"}
{"type": "sources", "sources": ["doc.pdf (Page 1)"]}
{"type": "done"}
```

See the [AI Service Streaming Protocol](../ai-service/README.md#streaming-protocol) for the full event schema.

---

## Services

### `PythonAIService`

A typed `HttpClient` service registered via dependency injection. It communicates with the Python FastAPI service.

**Key methods:**

| Method                        | HTTP Call                          | Returns                   |
| ----------------------------- | ---------------------------------- | ------------------------- |
| `StreamChatAsync(uid, q, m)`  | `POST /chat/stream`               | `IAsyncEnumerable<string>` (SSE lines) |
| `SaveFactAsync(uid, fact)`    | `POST /memory/save`               | JSON string               |
| `ListFactsAsync(uid)`         | `GET /memory/facts/{uid}`          | JSON string               |
| `DeleteFactAsync(uid, kw)`    | `DELETE /memory/facts/{uid}?kw=…`  | JSON string               |
| `WipeMemoryAsync(uid)`        | `POST /memory/wipe`               | void                      |
| `ClearBufferAsync(uid)`       | `POST /memory/clear-buffer`       | void                      |
| `UploadDocumentAsync(stream, name)` | `POST /documents/upload`    | JSON string               |
| `ListDocumentsAsync()`        | `GET /documents`                   | JSON string               |
| `RemoveDocumentAsync(name)`   | `DELETE /documents/{name}`         | JSON string               |

The `HttpClient` is configured with a 5-minute timeout to accommodate slow LLM responses.

---

## Database

The backend uses **SQLite** with **Entity Framework Core**. The database file (`app.db`) is auto-created on first startup via `EnsureCreated()`.

### Schema

The database contains only ASP.NET Core Identity tables:

| Table                    | Purpose                            |
| ------------------------ | ---------------------------------- |
| `AspNetUsers`            | User accounts (email, username, password hash) |
| `AspNetRoles`            | Role definitions (unused currently) |
| `AspNetUserRoles`        | User-role mappings                  |
| `AspNetUserClaims`       | Additional user claims              |
| `AspNetUserLogins`       | External login providers            |
| `AspNetUserTokens`       | Authentication tokens               |
| `AspNetRoleClaims`       | Role claim mappings                 |

> **Note:** All AI data (documents, facts, chat history) is stored in ChromaDB by the Python service, not in SQLite.

---

## Running Locally

**Prerequisites:** .NET 8 SDK, Python AI service running on port 8100.

```bash
cd backend

# Restore NuGet packages
dotnet restore

# Run (database auto-creates on first start)
dotnet run --urls http://localhost:5000
```

For development with hot-reload:
```bash
dotnet watch run --urls http://localhost:5000
```

**Verify:**
```bash
curl -X POST http://localhost:5000/api/auth/register \
  -H "Content-Type: application/json" \
  -d '{"username":"test","email":"test@test.com","password":"test123"}'
```

---

## Docker

```bash
docker build -t libraryai-backend ./backend
docker run -p 5000:5000 \
  -e PythonAI__BaseUrl=http://ai-service:8100 \
  -e Jwt__Key=your-production-secret-key-at-least-32-chars \
  -v backend-data:/data \
  libraryai-backend
```

---

## Troubleshooting

| Problem | Solution |
| ------- | -------- |
| `401 Unauthorized` on all requests | Check that the JWT token is included in the `Authorization: Bearer` header |
| SignalR connection fails | Verify the token is passed as `?access_token=` query parameter |
| `502 Bad Gateway` from AI calls | Ensure the Python AI service is running at the configured `PythonAI:BaseUrl` |
| Database locked errors | SQLite doesn't handle high concurrency well; this is fine for development |
| `Jwt:Key` warning | **Change the default key** before any production deployment |
