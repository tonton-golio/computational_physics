import { redirect } from "next/navigation";

// Redirect to graph-based navigation
export default function TopicPage() {
  redirect("/graph");
}
