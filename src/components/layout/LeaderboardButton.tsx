import Link from "next/link";
import { ICON_BUTTON_CLASS } from "@/components/ui/icon-button";
import { TrophyIcon } from "@/components/ui/icons";

export function LeaderboardButton() {
  return (
    <Link
      href="/leaderboard"
      className={ICON_BUTTON_CLASS}
      aria-label="Leaderboard"
      title="Leaderboard"
    >
      <TrophyIcon />
    </Link>
  );
}
