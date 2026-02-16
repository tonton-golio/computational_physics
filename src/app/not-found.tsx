import { Button } from "@/components/ui/button";

export default function NotFound() {
  return (
    <div className="min-h-screen bg-gray-50 flex items-center justify-center px-6">
      <div className="text-center">
        <div className="text-6xl mb-6">🐨</div>
        <h1 className="text-4xl font-bold text-gray-900 mb-4">Page Not Found</h1>
        <p className="text-lg text-gray-600 mb-8">
          The page you&apos;re looking for doesn&apos;t exist or has been moved.
        </p>
        <div className="flex gap-4 justify-center">
          <Button href="/graph" size="lg">
            Explore Topics
          </Button>
          <Button href="/about" variant="outline" size="lg">
            About
          </Button>
        </div>
      </div>
    </div>
  );
}
