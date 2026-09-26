import { NextResponse } from "next/server";

export const dynamic = "force-dynamic";

function publicIp(value: string | null) {
  const ip = value?.split(",")[0]?.trim() ?? "";
  if (!ip || ip === "::1" || ip.startsWith("127.") || ip.startsWith("10.") || ip.startsWith("192.168.")) return "";
  const parts = ip.split(".");
  if (parts.length === 4 && parts[0] === "172") {
    const second = Number(parts[1]);
    if (second >= 16 && second <= 31) return "";
  }
  return ip;
}

function langFor(country: string) {
  return country.toUpperCase() === "TR" ? "tr" : "en";
}

export async function GET(request: Request) {
  const headers = request.headers;
  const headerCountry =
    headers.get("x-vercel-ip-country") ||
    headers.get("cf-ipcountry") ||
    headers.get("x-country-code") ||
    "";

  if (headerCountry && headerCountry !== "XX") {
    return NextResponse.json({
      lang: langFor(headerCountry),
      country: headerCountry.toUpperCase(),
      source: "header",
    });
  }

  const ip = publicIp(headers.get("x-forwarded-for") || headers.get("x-real-ip"));
  const endpoint = ip ? `https://ipwho.is/${encodeURIComponent(ip)}` : "https://ipwho.is/";

  try {
    const response = await fetch(endpoint, { cache: "no-store" });
    if (!response.ok) throw new Error("geo lookup failed");
    const data = (await response.json()) as { success?: boolean; country_code?: string };
    const country = String(data.country_code || "").toUpperCase();
    if (!data.success || !country) throw new Error("geo lookup empty");
    return NextResponse.json({ lang: langFor(country), country, source: ip ? "forwarded" : "ipwho" });
  } catch {
    return NextResponse.json({ lang: "en", country: "", source: "fallback" });
  }
}
