import { useState, useEffect, useCallback, useRef } from "react";
import {
  HubConnectionBuilder,
  HubConnection,
  LogLevel,
} from "@microsoft/signalr";
import type { ChatEvent, Paper } from "../types";

interface Callbacks {
  onToken: (content: string) => void;
  onSources: (sources: string[]) => void;
  onStatus: (message: string) => void;
  onPapers: (papers: Paper[]) => void;
  onDone: () => void;
}

export function useSignalR(callbacks: Callbacks) {
  const [connection, setConnection] = useState<HubConnection | null>(null);
  const [isConnected, setIsConnected] = useState(false);
  const cbRef = useRef(callbacks);
  cbRef.current = callbacks;

  useEffect(() => {
    const token = localStorage.getItem("token");
    if (!token) return;

    const conn = new HubConnectionBuilder()
      .withUrl("/hubs/chat", { accessTokenFactory: () => token })
      .withAutomaticReconnect()
      .configureLogging(LogLevel.Warning)
      .build();

    // Allow long-running research queries without disconnect
    conn.serverTimeoutInMilliseconds = 10 * 60 * 1000;
    conn.keepAliveIntervalInMilliseconds = 15 * 1000;

    conn.on("ReceiveEvent", (raw: string) => {
      const e: ChatEvent = JSON.parse(raw);
      switch (e.type) {
        case "token":
          cbRef.current.onToken(e.content || "");
          break;
        case "sources":
          cbRef.current.onSources(e.sources || []);
          break;
        case "papers":
          console.log("[papers event]", e.papers);
          cbRef.current.onPapers(e.papers || []);
          break;
        case "status":
          cbRef.current.onStatus(e.message || "");
          break;
        case "done":
          cbRef.current.onDone();
          break;
      }
    });

    conn.onreconnecting(() => setIsConnected(false));
    conn.onreconnected(() => setIsConnected(true));
    conn.onclose(() => setIsConnected(false));

    conn.start().then(() => {
      setConnection(conn);
      setIsConnected(true);
    });

    return () => {
      conn.stop();
    };
  }, []);

  const send = useCallback(
    async (message: string, mode: string) => {
      if (!connection) throw new Error("Not connected");
      await connection.invoke("SendMessage", message, mode);
    },
    [connection]
  );

  return { isConnected, send };
}
