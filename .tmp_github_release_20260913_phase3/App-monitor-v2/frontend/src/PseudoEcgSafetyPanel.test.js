import test from "node:test";
import assert from "node:assert/strict";
import React from "react";
import { renderToStaticMarkup } from "react-dom/server";

import PseudoEcgSafetyPanel from "./PseudoEcgSafetyPanel.js";

test("疑似ECGの警告、測定禁止、品質、棄却理由を常時表示する", () => {
  const html = renderToStaticMarkup(React.createElement(PseudoEcgSafetyPanel, {
    frame: {
      generation_accepted: false,
      ppg_sqi: 0.2,
      signal_coverage: 0.5,
      generation_abstention_reason: "insufficient_valid_samples",
      latest_pulse_interval_ms: 810,
      interval_cv: 0.12,
      rmssd_ms: 70,
      estimated_pat_ms: 200,
      display_delay_ms: 500,
      input_timestamp_start_seconds: 100,
      display_timestamp_start_seconds: 99.5,
      generation_mode: "population_prior",
      morphology_source: "population_mean_latent",
      timing_confidence: 0.9,
      morphology_confidence: 0.0,
      repolarization_duration_prior_ms: 380,
      p_wave_generated: false,
      qt_measurement_supported: false,
      pq_pr_measurement_supported: false,
    },
    protocolIssue: "sequence_gap",
  }));
  assert.match(html, /PPG由来疑似ECG・診断用ではない/);
  assert.match(html, /P波、PR、QRS幅、QT、STの測定には使用できません/);
  assert.match(html, /SQI: 0.200/);
  assert.match(html, /生成可否: 生成停止/);
  assert.match(html, /insufficient_valid_samples/);
  assert.match(html, /推定PAT: 200.0 ms/);
  assert.match(html, /再分極表示prior: 380.0 ms（QTではない）/);
  assert.match(html, /P波生成: なし/);
  assert.match(html, /sequence_gap/);
});
