import type { Metadata } from "next";
import "./globals.css";

export const metadata: Metadata = { title: "SRE Copilot", description: "Chat, approve and review SRE actions" };

export default function RootLayout({ children }: { children: React.ReactNode }) {
  return (
    <html lang="en">
      <body>{children}</body>
    </html>
  );
}
