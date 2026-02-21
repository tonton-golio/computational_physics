import { describe, expect, it } from "vitest";
import { simulationTitle, SIMULATION_DESCRIPTIONS } from "./simulation-descriptions";

describe("simulation-descriptions", () => {
  describe("simulationTitle", () => {
    it("converts kebab-case to Title Case", () => {
      expect(simulationTitle("steepest-descent")).toBe("Steepest Descent");
    });

    it("handles single word", () => {
      expect(simulationTitle("percolation")).toBe("Percolation");
    });

    it("strips adl- prefix", () => {
      expect(simulationTitle("adl-vae-latent")).toBe("Vae Latent");
    });

    it("strips aml- prefix", () => {
      expect(simulationTitle("aml-gradient-descent")).toBe("Gradient Descent");
    });

    it("does not strip other prefixes", () => {
      expect(simulationTitle("orl-something")).toBe("Orl Something");
    });

    it("handles multiple hyphens", () => {
      expect(simulationTitle("a-b-c-d")).toBe("A B C D");
    });
  });

  describe("SIMULATION_DESCRIPTIONS", () => {
    it("is a non-empty record", () => {
      expect(Object.keys(SIMULATION_DESCRIPTIONS).length).toBeGreaterThan(0);
    });

    it("all values are non-empty strings", () => {
      for (const [id, desc] of Object.entries(SIMULATION_DESCRIPTIONS)) {
        expect(typeof desc, `${id} description`).toBe("string");
        expect(desc.length, `${id} should not be empty`).toBeGreaterThan(0);
      }
    });
  });
});
