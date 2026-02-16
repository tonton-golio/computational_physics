import { redirect } from "next/navigation";

// Redirect to graph-based navigation
export default function TopicsPage() {
  redirect("/graph");
}
