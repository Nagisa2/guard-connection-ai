import React from "react";

function metric(label, value) {
  return React.createElement("span", { key: label }, `${label}: ${value}`);
}

export default function PseudoEcgSafetyPanel({ frame, protocolIssue }) {
  if (!frame) return null;
  return React.createElement(
    "section",
    { "data-testid": "pseudo-ecg-safety-panel" },
    React.createElement(
      "div",
      {
        style: {
          background: "#fff7ed",
          border: "1px solid #f59e0b",
          color: "#9a3412",
          padding: "10px 12px",
          borderRadius: "8px",
          marginBottom: "10px",
          fontSize: "12px",
          fontWeight: "bold",
        },
      },
      "PPG由来疑似ECG・診断用ではない。P波、PR、QRS幅、QT、STの測定には使用できません。",
    ),
    React.createElement(
      "div",
      {
        style: {
          marginBottom: "10px",
          display: "grid",
          gridTemplateColumns: "repeat(auto-fit, minmax(160px, 1fr))",
          gap: "8px",
          fontSize: "12px",
        },
      },
      metric("SQI", Number(frame.ppg_sqi).toFixed(3)),
      metric("coverage", Number(frame.signal_coverage).toFixed(3)),
      metric("生成可否", frame.generation_accepted ? "生成可能" : "生成停止"),
      metric("生成停止理由", frame.generation_abstention_reason || "なし"),
      metric("pulse interval", frame.latest_pulse_interval_ms == null ? "未算出" : `${Number(frame.latest_pulse_interval_ms).toFixed(1)} ms`),
      metric("interval CV", frame.interval_cv == null ? "未算出" : Number(frame.interval_cv).toFixed(3)),
      metric("RMSSD", frame.rmssd_ms == null ? "未算出" : `${Number(frame.rmssd_ms).toFixed(1)} ms`),
      metric("生成方式", frame.generation_mode),
      metric("形態source", frame.morphology_source),
      metric("timing confidence", frame.timing_confidence == null ? "未算出" : Number(frame.timing_confidence).toFixed(3)),
      metric("morphology confidence", frame.morphology_confidence == null ? "未算出" : Number(frame.morphology_confidence).toFixed(3)),
      metric("再分極表示prior", frame.repolarization_duration_prior_ms == null ? "未算出" : `${Number(frame.repolarization_duration_prior_ms).toFixed(1)} ms（QTではない）`),
      metric("P波生成", frame.p_wave_generated ? "あり" : "なし"),
      metric("推定PAT", frame.estimated_pat_ms == null ? "未算出" : `${Number(frame.estimated_pat_ms).toFixed(1)} ms`),
      metric("表示遅延", `${frame.display_delay_ms} ms`),
      metric("入力時刻", `${Number(frame.input_timestamp_start_seconds).toFixed(3)} s`),
      metric("表示時刻", `${Number(frame.display_timestamp_start_seconds).toFixed(3)} s`),
      metric("受信状態", protocolIssue || "連番正常"),
    ),
  );
}
