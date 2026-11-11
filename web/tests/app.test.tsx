import { fireEvent, render, screen, waitFor } from "@testing-library/react";
import { afterEach, describe, expect, it, vi } from "vitest";
import { App } from "../src/App";

function reply(data: unknown, status = 200) {
  return Promise.resolve({
    ok: status >= 200 && status < 300,
    status,
    json: async () => data,
  } as Response);
}

describe("LLM Fine-tuning and Serving workbench", () => {
  afterEach(() => vi.unstubAllGlobals());

  it("shows an evidence empty state and renders live model output", async () => {
    const fetchMock = vi.fn(async (input: RequestInfo | URL) => {
      const url = String(input);
      if (url.endsWith("/health"))
        return reply({
          status: "ok",
          finetuning_regression_tester: { available: false },
          aftertune: { available: false },
        });
      if (url.endsWith("/models"))
        return reply({
          base: "local-test-model",
          adapter_status: { state: "unavailable" },
        });
      if (url.endsWith("/experiments")) return reply([]);
      if (url.endsWith("/datasets/summary")) return reply({}, 404);
      if (url.endsWith("/predict"))
        return reply({
          ticket: { intent: "card_arrival", priority: "unspecified" },
          model_id: "local-test-model",
          adapter_id: null,
          model_revision: "abc123",
          latency_ms: 12,
          warnings: [],
          field_sources: { intent: "model" },
          raw_output: '{"intent":"card_arrival"}',
          output_tokens: 5,
          invalid_label: false,
        });
      return reply({ detail: "not found" }, 404);
    });
    vi.stubGlobal("fetch", fetchMock);
    render(<App />);
    expect(
      await screen.findByText(/No experiment runs registered yet/),
    ).toBeTruthy();
    fireEvent.click(screen.getByRole("button", { name: /Run inference/ }));
    expect(
      await screen.findByText(
        (_, element) =>
          element?.tagName === "PRE" &&
          element.textContent?.includes('"intent": "card_arrival"') === true,
      ),
    ).toBeTruthy();
    await waitFor(() => expect(fetchMock).toHaveBeenCalled());
  });
});
