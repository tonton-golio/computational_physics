import { redirect } from "next/navigation";

// Redirect home to graph-based navigation
export default function Home() {
  redirect("/graph");
}
