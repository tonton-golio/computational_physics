import Link from "next/link";
import Antigravity from "@/components/effects/Antigravity";
import { TOPIC_ROUTES } from "@/lib/topic-navigation";
import { getLessonsForTopic } from "@/lib/topic-navigation.server";

export default function HomePage() {
  const topicCount = TOPIC_ROUTES.length;
  const subtopicCount = TOPIC_ROUTES.reduce(
    (sum, route) => sum + getLessonsForTopic(route.contentId).length,
    0
  );
  return (
    <div className="relative h-[calc(100vh-4rem)] overflow-hidden">
      <div className="absolute inset-0 flex justify-center">
        <div className="h-full w-full">
          <Antigravity
            count={320}
            magnetRadius={6}
            ringRadius={7}
            waveSpeed={0.4}
            waveAmplitude={1}
            particleSize={1.5}
            particleVariance={1}
            lerpSpeed={0.05}
            color="#B19EEF"
            autoAnimate
            rotationSpeed={0}
            depthFactor={1}
            pulseSpeed={3}
            particleShape="capsule"
            fieldStrength={10}
          />
        </div>
      </div>

      <div className="pointer-events-none absolute inset-0 bg-[var(--background)]/30" />

      <div className="relative z-10 h-full px-[50px] py-8">
        <div className="mx-auto grid h-full max-w-[1200px] content-center grid-cols-1 gap-4 lg:grid-cols-12">
          <section className="lg:col-span-7 flex flex-col justify-center rounded-lg border border-[var(--border-strong)] bg-[var(--surface-1)]/30 p-5 backdrop-blur-lg">
            <p className="font-mono text-xs uppercase tracking-[0.28em] text-[var(--accent)]">koala brain :: v2</p>
            <p className="mt-3 max-w-2xl text-sm text-[var(--text-muted)]">
              Computational physics you can touch. Every equation runs, every model bends, every
              simulation is yours to break.
            </p>
            <p className="mt-2 max-w-2xl text-xs text-[var(--text-soft)]">
              10 topics spanning quantum optics to deep learning — each one ships with derivations,
              live simulations, and code you can fork. Theory without practice is just LaTeX.
            </p>

            <div className="mt-7 flex flex-wrap items-center gap-3 font-mono text-sm">
              <Link
                href="/topics"
                className="rounded-md border border-[var(--accent)] bg-[var(--surface-2)] px-4 py-2 text-[var(--accent)] transition hover:bg-[var(--accent)] hover:text-black"
              >
                ./start --topics
              </Link>
              <a
                href="https://github.com/tonton-golio/computational_physics"
                target="_blank"
                rel="noopener noreferrer"
                className="rounded-md border border-[var(--border-strong)] px-4 py-2 text-[var(--text-muted)] transition hover:text-[var(--text-strong)]"
              >
                git clone
              </a>
            </div>
          </section>

          <section className="lg:col-span-5 flex min-h-0 flex-col gap-3 rounded-lg border border-[var(--border-strong)] bg-[var(--surface-1)]/30 p-4 font-mono text-sm backdrop-blur-lg">
            <div className="rounded-md border border-[var(--border-strong)] bg-[var(--surface-2)]/40 dark:bg-black/30 p-3">
              <p className="text-[var(--text-soft)]">$ status</p>
              <p className="mt-1 text-[var(--text-strong)]">{topicCount} topics, {subtopicCount} subtopics — all systems nominal.</p>
              <p className="mt-1 text-[var(--text-soft)]">$ mission</p>
              <p className="mt-1 text-[var(--text-strong)]">An open, self-evolving physics lab — powered by agents, shaped by contributors.</p>
              <p className="mt-1 text-[var(--text-soft)]">$ discord</p>
              <p className="mt-1 text-[var(--text-strong)]"><a href="https://discord.gg/JzZRhUNV5c" target="_blank" rel="noopener noreferrer" className="text-[var(--accent)] underline underline-offset-2 hover:text-[var(--text-strong)] transition">Join the community →</a></p>
            </div>

            <div className="rounded-md border border-[var(--border-strong)] p-2.5 text-xs">
              <p className="text-[var(--text-soft)]">Engine & Growth</p>
              <p className="mt-1 text-[var(--text-strong)]">AI agents author, validate, and publish new content continuously — filling gaps in the knowledge graph as they surface. The platform doesn&apos;t just host material, it cultivates it. New subtopics emerge, existing ones sharpen, and the whole system quietly compounds while you&apos;re busy studying.</p>
            </div>
          </section>
        </div>
      </div>
    </div>
  );
}
