"use client";

import Link from "next/link";
import { useCurrentUser } from "@/lib/use-current-user";
import { ICON_BUTTON_CLASS } from "@/components/ui/icon-button";
import { UserIcon } from "@/components/ui/icons";

export function AuthButton() {
  const user = useCurrentUser();

  return (
    <Link
      href={user ? "/profile" : "/login"}
      className={ICON_BUTTON_CLASS}
      aria-label={user ? "Profile" : "Sign in"}
    >
      <UserIcon />
    </Link>
  );
}
