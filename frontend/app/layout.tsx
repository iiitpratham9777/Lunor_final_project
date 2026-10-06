import type { ReactNode } from "react";

import "./globals.css";

export const metadata = {
  title: "Multimodal AI Investigation Agent",
  description: "Planner Executor Verifier dashboard for anomaly investigation",
};

export default function RootLayout({ children }: { children: ReactNode }) {
  return (
    <html lang="en">
      <body>{children}</body>
    </html>
  );
}
