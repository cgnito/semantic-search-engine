import "./globals.css";

export const metadata = {
  title: "Abdul / Semantic Search",
  description: "Tweet Explorer",
};

export default function RootLayout({
  children,
}: {
  children: React.ReactNode;
}) {
  return (
    <html lang="en" className="dark">
      <body className="antialiased">{children}</body>
    </html>
  );
}