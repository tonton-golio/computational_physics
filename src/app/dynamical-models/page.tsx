import { MarkdownContent } from '@/components/content/MarkdownContent';

const content = `# Dynamical Models



# Title
Dynamical Models in Molecular Biology

# Welcome to DynBio!
- The course introduces the basic knowledge and skills of biology, physics, and 
mathematics required for a modern, integrated understanding of dynamical biological 
systems.
- We focus on simple systems, often bacterial examples, but look for general rules that are relevant for many biological systems.
- We focus on finding the overall logic, not the details

# Aim of the course: BRIDGE THE GAP
- Make future communication across the fields easier – find friends!
- Get used to the different "languages", and understand why they are used.
- Learn the basics of your non-specialty fields, and learn how to obtain more information if necessary. 
- Learn new things within your field and get a new perspective on the whole thing.
`;

export default function DynamicalModelsPage() {
  return (
    <div className="min-h-screen bg-[#0a0a15] text-white">
      <div className="container mx-auto px-4 py-8">
        <MarkdownContent content={content} />
      </div>
    </div>
  );
}