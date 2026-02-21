import { describe, expect, it } from "vitest";
import { markdownToHtml } from "./markdown-to-html";

const noKatex = null;

describe("markdownToHtml", () => {
  describe("headers", () => {
    it("converts h1", () => {
      const { html } = markdownToHtml("# Hello World", noKatex);
      expect(html).toContain("<h1");
      expect(html).toContain("Hello World");
    });

    it("converts h2", () => {
      const { html } = markdownToHtml("## Section", noKatex);
      expect(html).toContain("<h2");
      expect(html).toContain("Section");
    });

    it("converts h3", () => {
      const { html } = markdownToHtml("### Subsection", noKatex);
      expect(html).toContain("<h3");
    });

    it("converts h4 through h6", () => {
      expect(markdownToHtml("#### H4", noKatex).html).toContain("<h4");
      expect(markdownToHtml("##### H5", noKatex).html).toContain("<h5");
      expect(markdownToHtml("###### H6", noKatex).html).toContain("<h6");
    });

    it("does not convert mid-line hashes", () => {
      const { html } = markdownToHtml("some text # not a header", noKatex);
      expect(html).not.toContain("<h1");
    });
  });

  describe("inline formatting", () => {
    it("converts bold with **", () => {
      const { html } = markdownToHtml("**bold text**", noKatex);
      expect(html).toContain("<strong>bold text</strong>");
    });

    it("converts bold with __", () => {
      const { html } = markdownToHtml("__bold text__", noKatex);
      expect(html).toContain("<strong>bold text</strong>");
    });

    it("converts italic with *", () => {
      const { html } = markdownToHtml("*italic text*", noKatex);
      expect(html).toContain("<em>italic text</em>");
    });

    it("converts inline code", () => {
      const { html } = markdownToHtml("`const x = 1`", noKatex);
      expect(html).toContain("<code");
      expect(html).toContain("const x = 1");
    });
  });

  describe("links", () => {
    it("converts markdown links to anchor tags", () => {
      const { html } = markdownToHtml("[Google](https://google.com)", noKatex);
      expect(html).toContain('href="https://google.com"');
      expect(html).toContain("Google");
      expect(html).toContain('target="_blank"');
      expect(html).toContain('rel="noopener noreferrer"');
    });
  });

  describe("code blocks", () => {
    it("converts fenced code blocks", () => {
      const md = "```python\nprint('hello')\n```";
      const { html } = markdownToHtml(md, noKatex);
      expect(html).toContain("<pre");
      expect(html).toContain("<code>");
      expect(html).toContain("print('hello')");
    });

    it("handles code blocks without language specifier", () => {
      const md = "```\nsome code\n```";
      const { html } = markdownToHtml(md, noKatex);
      expect(html).toContain("<pre");
      expect(html).toContain("some code");
    });
  });

  describe("lists", () => {
    it("converts unordered list items", () => {
      const md = "- Item one\n- Item two";
      const { html } = markdownToHtml(md, noKatex);
      expect(html).toContain("<li");
      expect(html).toContain("Item one");
      expect(html).toContain("Item two");
      expect(html).toContain("<ul");
    });

    it("converts numbered list items", () => {
      const md = "1. First\n2. Second";
      const { html } = markdownToHtml(md, noKatex);
      expect(html).toContain("<li");
      expect(html).toContain("First");
      expect(html).toContain("Second");
    });
  });

  describe("horizontal rule", () => {
    it("converts --- to <hr>", () => {
      const { html } = markdownToHtml("---", noKatex);
      expect(html).toContain("<hr");
    });
  });

  describe("tables", () => {
    it("converts markdown tables", () => {
      const md = "| Col A | Col B |\n| --- | --- |\n| 1 | 2 |";
      const { html } = markdownToHtml(md, noKatex);
      expect(html).toContain("<table");
      expect(html).toContain("<tr>");
      expect(html).toContain("Col A");
    });

    it("promotes first row to th", () => {
      const md = "| Header |\n| --- |\n| data |";
      const { html } = markdownToHtml(md, noKatex);
      expect(html).toContain("<th");
    });
  });

  describe("paragraphs", () => {
    it("wraps plain text in <p> tags", () => {
      const { html } = markdownToHtml("Hello\n\nWorld", noKatex);
      expect(html).toContain("<p");
      expect(html).toContain("Hello");
      expect(html).toContain("World");
    });
  });

  describe("LaTeX (without KaTeX module)", () => {
    it("preserves inline math content when katex is null", () => {
      const { html } = markdownToHtml("The formula $E=mc^2$ is famous", noKatex);
      expect(html).toContain("katex-inline");
      expect(html).toContain("E=mc^2");
    });

    it("preserves block math content when katex is null", () => {
      const { html } = markdownToHtml("$$\nE = mc^2\n$$", noKatex);
      expect(html).toContain("katex-block");
      expect(html).toContain("E = mc^2");
    });

    it("handles \\( \\) inline syntax", () => {
      const { html } = markdownToHtml("Inline \\(x^2\\) math", noKatex);
      expect(html).toContain("katex-inline");
      expect(html).toContain("x^2");
    });

    it("handles \\[ \\] block syntax", () => {
      const { html } = markdownToHtml("\\[\nx^2\n\\]", noKatex);
      expect(html).toContain("katex-block");
    });
  });

  describe("LaTeX (with mock KaTeX)", () => {
    const mockKatex = {
      renderToString(math: string, opts?: { displayMode?: boolean }) {
        return `<span class="mock-katex" data-display="${opts?.displayMode}">${math}</span>`;
      },
    };

    it("renders inline math via katex module", () => {
      const { html } = markdownToHtml("$x^2$", mockKatex);
      expect(html).toContain("mock-katex");
      expect(html).toContain("x^2");
    });

    it("renders block math via katex module", () => {
      const { html } = markdownToHtml("$$\nF=ma\n$$", mockKatex);
      expect(html).toContain('data-display="true"');
      expect(html).toContain("F=ma");
    });
  });

  describe("placeholders", () => {
    it("extracts simulation placeholders", () => {
      const { html, placeholders } = markdownToHtml(
        "Before\n\n[[simulation percolation-demo]]\n\nAfter",
        noKatex
      );
      expect(placeholders).toHaveLength(1);
      expect(placeholders[0].type).toBe("simulation");
      expect(placeholders[0].id).toBe("percolation-demo");
      expect(placeholders[0].index).toBe(0);
      expect(html).toContain("%%PLACEHOLDER_0%%");
    });

    it("extracts figure placeholders", () => {
      const { placeholders } = markdownToHtml("[[figure my-figure]]", noKatex);
      expect(placeholders).toHaveLength(1);
      expect(placeholders[0].type).toBe("figure");
      expect(placeholders[0].id).toBe("my-figure");
    });

    it("extracts code-editor placeholders", () => {
      const { placeholders } = markdownToHtml(
        "[[code-editor python|pseudo]]",
        noKatex
      );
      expect(placeholders).toHaveLength(1);
      expect(placeholders[0].type).toBe("code-editor");
    });

    it("handles multiple placeholders with sequential indices", () => {
      const md =
        "[[simulation sim-a]]\n\n[[simulation sim-b]]\n\n[[figure fig-1]]";
      const { placeholders } = markdownToHtml(md, noKatex);
      expect(placeholders).toHaveLength(3);
      expect(placeholders[0].index).toBe(0);
      expect(placeholders[1].index).toBe(1);
      expect(placeholders[2].index).toBe(2);
    });

    it("returns empty placeholders when none present", () => {
      const { placeholders } = markdownToHtml("Just plain text.", noKatex);
      expect(placeholders).toEqual([]);
    });
  });

  describe("code-toggle blocks", () => {
    it("extracts code-toggle from paired fenced blocks", () => {
      const md = [
        "```python",
        "x = 1",
        "```",
        "<!--code-toggle-->",
        "```pseudocode",
        "set x to 1",
        "```",
      ].join("\n");
      const { placeholders } = markdownToHtml(md, noKatex);
      expect(placeholders).toHaveLength(1);
      expect(placeholders[0].type).toBe("code-toggle");
      expect(placeholders[0].id).toContain("x = 1");
      expect(placeholders[0].id).toContain("|||");
      expect(placeholders[0].id).toContain("set x to 1");
    });
  });

  describe("static mode", () => {
    it("renders simulation as static placeholder in static mode", () => {
      const { html, placeholders } = markdownToHtml(
        "[[simulation steepest-descent]]",
        noKatex,
        { staticMode: true }
      );
      expect(placeholders).toHaveLength(0);
      expect(html).toContain("Interactive Simulation");
      expect(html).toContain("Steepest Descent");
    });

    it("renders code-editor as static pre block", () => {
      const { html } = markdownToHtml("[[code-editor some code]]", noKatex, {
        staticMode: true,
      });
      expect(html).toContain("<pre");
      expect(html).toContain("some code");
    });
  });

  describe("figures", () => {
    it("returns empty figures array in non-static mode", () => {
      const { figures } = markdownToHtml("[[figure test]]", noKatex);
      expect(figures).toEqual([]);
    });
  });
});
