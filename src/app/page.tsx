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
    <div className="relative h-[calc(100vh-4rem)] overflow-hidden bg-[var(--background)]">
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

      <div className="pointer-events-none absolute inset-0 bg-[var(--background)]/62" />

      <div className="relative z-10 h-full px-[50px] py-8">
        <div className="mx-auto grid h-full max-w-[1200px] content-center grid-cols-1 gap-4 lg:grid-cols-12">
          <section className="lg:col-span-7 flex flex-col justify-center rounded-lg border border-[var(--border-strong)] bg-[var(--surface-1)]/55 p-5 backdrop-blur-sm">
            <p className="font-mono text-xs uppercase tracking-[0.28em] text-[var(--accent)]">koala brain :: v2</p>
            <p className="mt-3 max-w-2xl text-sm text-[var(--text-muted)]">
              Less passive reading. More executable math. Every topic is designed to be tested,
              visualized, and hacked.
            </p>
            <p className="mt-2 max-w-2xl text-xs text-[var(--text-soft)]">
              From quantum optics to inverse problems, each module blends derivations, simulation
              controls, and implementation notes so theory and practice stay in the same loop.
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

          <section className="lg:col-span-5 flex min-h-0 flex-col gap-3 rounded-lg border border-[var(--border-strong)] bg-[var(--surface-1)]/60 p-4 font-mono text-sm backdrop-blur-sm">
            <div className="rounded-md border border-[var(--border-strong)] bg-black/30 p-3">
              <p className="text-[var(--text-soft)]">$ status</p>
              <p className="mt-1 text-[var(--text-strong)]">{topicCount} topics, {subtopicCount} subtopics — all systems nominal.</p>
              <p className="mt-1 text-[var(--text-soft)]">$ mission</p>
              <p className="mt-1 text-[var(--text-strong)]">Build a self-growing, agent-managed learning platform.</p>
              <p className="mt-1 text-[var(--text-soft)]">$ discord</p>
              <p className="mt-1 text-[var(--text-strong)]"><a href="https://discord.gg/JzZRhUNV5c" target="_blank" rel="noopener noreferrer" className="text-[var(--accent)] underline underline-offset-2 hover:text-[var(--text-strong)] transition">Join the community →</a></p>
            </div>

            <div className="grid grid-cols-2 gap-2 text-xs">
              <div className="rounded-md border border-[var(--border-strong)] p-2.5">
                <p className="text-[var(--text-soft)]">Engine</p>
                <p className="mt-1 text-[var(--text-strong)]">AI agent-driven content</p>
              </div>
              <div className="rounded-md border border-[var(--border-strong)] p-2.5">
                <p className="text-[var(--text-soft)]">Growth</p>
                <p className="mt-1 text-[var(--text-strong)]">Self-expanding knowledge graph</p>
              </div>
              <div className="rounded-md border border-[var(--border-strong)] p-2.5">
                <p className="text-[var(--text-soft)]">Stack</p>
                <p className="mt-1 text-[var(--text-strong)]">Next.js, TypeScript, Plotly</p>
              </div>
              <div className="rounded-md border border-[var(--border-strong)] p-2.5">
                <p className="text-[var(--text-soft)]">Cycle</p>
                <p className="mt-1 text-[var(--text-strong)]">Generate → validate → publish</p>
              </div>
            </div>
          </section>
        </div>
      </div>
    </div>
  );
}
