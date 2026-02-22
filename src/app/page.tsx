import Link from "next/link";
import { TOPIC_ROUTES } from "@/lib/topic-navigation";
import AntigravityClient from "@/components/effects/AntigravityClient";
import { getLessonsForTopic } from "@/features/content/topic-lessons";

export default function HomePage() {
  const topicCount = TOPIC_ROUTES.length;
  const subtopicCount = TOPIC_ROUTES.reduce(
    (sum, route) => sum + getLessonsForTopic(route.topicId).length,
    0
  );
  return (
    <div className="relative h-[calc(100vh-4rem)] overflow-hidden">
      <div className="absolute inset-0 flex justify-center">
        <div className="h-full w-full">
          <AntigravityClient
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

      <div className="relative z-10 h-full px-4 sm:px-8 lg:px-[50px] py-8">
        <div className="mx-auto grid h-full max-w-[1200px] content-center grid-cols-1 gap-4 lg:grid-cols-12">
          <section className="lg:col-span-7 flex flex-col justify-center rounded-lg border border-[var(--border-strong)] bg-[var(--surface-1)]/30 p-5 backdrop-blur-lg">
            <p className="font-mono text-xs uppercase tracking-[0.28em] text-[var(--accent)]">koala brain :: V02.01.002</p>
            <p className="mt-3 max-w-2xl text-sm text-[var(--text-muted)]">
              Computational physics you can touch. Every equation runs, every model bends, every
              simulation is yours to break.
            </p>
            <p className="mt-2 max-w-2xl text-xs text-[var(--text-soft)]">
              Community suggestions steer what gets built. AI agents author, validate, and publish new content — a new subject lands every week. Existing topics sharpen continuously while the knowledge graph quietly compounds. Theory without practice is just LaTeX.
            </p>

            <div className="mt-7 flex flex-wrap items-center gap-3 font-mono text-sm">
              <Link
                href="/topics"
                className="rounded-md border border-[var(--accent)] bg-[var(--surface-2)] px-4 py-2 text-[var(--accent)] transition hover:bg-[var(--accent)] hover:text-black"
              >
                enter
              </Link>
              <a
                href="https://github.com/tonton-golio/computational_physics"
                target="_blank"
                rel="noopener noreferrer"
                className="rounded-md border border-[var(--border-strong)] p-2 text-[var(--text-muted)] transition hover:text-[var(--text-strong)]"
                aria-label="GitHub"
              >
                <svg width="20" height="20" viewBox="0 0 24 24" fill="currentColor"><path d="M12 0C5.37 0 0 5.37 0 12c0 5.31 3.435 9.795 8.205 11.385.6.105.825-.255.825-.57 0-.285-.015-1.23-.015-2.235-3.015.555-3.795-.735-4.035-1.41-.135-.345-.72-1.41-1.23-1.695-.42-.225-1.02-.78-.015-.795.945-.015 1.62.87 1.845 1.23 1.08 1.815 2.805 1.305 3.495.99.105-.78.42-1.305.765-1.605-2.67-.3-5.46-1.335-5.46-5.925 0-1.305.465-2.385 1.23-3.225-.12-.3-.54-1.53.12-3.18 0 0 1.005-.315 3.3 1.23.96-.27 1.98-.405 3-.405s2.04.135 3 .405c2.295-1.56 3.3-1.23 3.3-1.23.66 1.65.24 2.88.12 3.18.765.84 1.23 1.905 1.23 3.225 0 4.605-2.805 5.625-5.475 5.925.435.375.81 1.095.81 2.22 0 1.605-.015 2.895-.015 3.3 0 .315.225.69.825.57A12.02 12.02 0 0 0 24 12c0-6.63-5.37-12-12-12Z"/></svg>
              </a>
              <a
                href="https://x.com/koaborain"
                target="_blank"
                rel="noopener noreferrer"
                className="rounded-md border border-[var(--border-strong)] p-2 text-[var(--text-muted)] transition hover:text-[var(--text-strong)]"
                aria-label="X"
              >
                <svg width="20" height="20" viewBox="0 0 24 24" fill="currentColor"><path d="M18.244 2.25h3.308l-7.227 8.26 8.502 11.24H16.17l-5.214-6.817L4.99 21.75H1.68l7.73-8.835L1.254 2.25H8.08l4.713 6.231zm-1.161 17.52h1.833L7.084 4.126H5.117z"/></svg>
              </a>
            </div>
          </section>

          <section className="lg:col-span-5 flex min-h-0 flex-col gap-3 rounded-lg border border-[var(--border-strong)] bg-[var(--surface-1)]/30 p-4 font-mono text-sm backdrop-blur-lg">
            <div className="rounded-md border border-[var(--border-strong)] bg-[var(--surface-2)]/40 dark:bg-black/30 p-3">
              <p className="text-[var(--text-soft)]">$ status</p>
              <p className="mt-1 text-[var(--text-strong)]">{topicCount} topics, {subtopicCount} subtopics.</p>
              <p className="mt-1 text-[var(--text-soft)]">$ mission</p>
              <p className="mt-1 text-[var(--text-strong)]">An open, self-evolving learning platform — powered by agents, shaped by contributors.</p>
            </div>
          </section>
        </div>
      </div>
    </div>
  );
}
