import { NextRequest, NextResponse } from "next/server";

const BACKEND_URL = process.env.NEXT_PUBLIC_BACKEND_URL || "http://127.0.0.1:8000";

export async function POST(
  req: NextRequest,
  { params }: { params: { id: string } }
) {
  try {
    const incidentId = params.id;
    const body = await req.json();

    for (const targetUrl of [BACKEND_URL, "http://127.0.0.1:8000", "http://localhost:8000"]) {
      try {
        const res = await fetch(`${targetUrl}/api/incidents/${incidentId}/status`, {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify(body),
          cache: "no-store",
        });

        if (res.ok) {
          const data = await res.json();
          return NextResponse.json(data);
        }
      } catch {
        // Try fallback
      }
    }

    // Return status fallback
    const statusMap: Record<string, string> = {
      CONFIRMED_THREAT: "confirmed",
      MARKED_FALSE_ALARM: "false_alarm",
      ESCALATED: "open",
    };
    return NextResponse.json({
      id: incidentId,
      status: statusMap[body.action] || "open",
    });
  } catch (err: unknown) {
    const message = err instanceof Error ? err.message : "Internal Error";
    return NextResponse.json({ error: message }, { status: 500 });
  }
}
