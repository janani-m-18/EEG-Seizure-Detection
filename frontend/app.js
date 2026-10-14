/**
 * Explainable CNN-BiLSTM EEG Signal Analysis System
 * Frontend Application & Real-Time Replay Controller
 * 
 * Research Prototype — Not for Clinical Diagnostic Use
 */

(() => {
  'use strict';

  const apiBase = window.EEG_API_BASE_URL || '';

  // DOM Elements - Navigation & Status
  const connectionStatus = document.querySelector('#connection-status');
  const statusText = document.querySelector('#status-text');

  // DOM Elements - Upload
  const fileInput = document.querySelector('#edf-file');
  const dropZone = document.querySelector('#drop-zone');
  const browseButton = document.querySelector('#browse-button');
  const fileInfoRow = document.querySelector('#file-info-row');
  const fileName = document.querySelector('#file-name');
  const fileSize = document.querySelector('#file-size');
  const analyzeButton = document.querySelector('#analyze-button');
  const progressBarContainer = document.querySelector('#progress-bar-container');
  const progressBar = document.querySelector('#progress-bar');
  const errorMessage = document.querySelector('#error-message');

  // DOM Elements - Results Dashboard
  const resultsDashboard = document.querySelector('#results');
  const resultCard = document.querySelector('.result-card');
  const predictionBadge = document.querySelector('#prediction-badge');
  const qualityBadge = document.querySelector('#quality-badge');

  // Concise Metrics
  const highestProbValue = document.querySelector('#highest-prob-value');
  const positiveWindowsValue = document.querySelector('#positive-windows-value');
  const eventsCountValue = document.querySelector('#events-count-value');

  // Optional Metric DOM Elements (Guarded)
  const longestRunValue = document.querySelector('#longest-run-value');
  const runDurationSub = document.querySelector('#run-duration-sub');
  const positiveRatioSub = document.querySelector('#positive-ratio-sub');
  const aggregationRuleText = document.querySelector('#aggregation-rule-text');
  const thresholdText = document.querySelector('#threshold-text');
  const requiredPersistenceText = document.querySelector('#required-persistence-text');
  const eventDetectedFlag = document.querySelector('#event-detected-flag');
  const eventDetectedSub = document.querySelector('#event-detected-sub');
  const thresholdValBox = document.querySelector('#threshold-val-box');
  const persistenceValBox = document.querySelector('#persistence-val-box');
  const meanProbValue = document.querySelector('#mean-prob-value');

  // Metadata ribbon (Guarded)
  const durationVal = document.querySelector('#duration-val');
  const samplingVal = document.querySelector('#sampling-val');
  const channelsVal = document.querySelector('#channels-val');
  const totalWindowsVal = document.querySelector('#total-windows-val');

  // Events Table
  const eventsSection = document.querySelector('#events-section');
  const eventsBadgeCount = document.querySelector('#events-badge-count');
  const eventsTableBody = document.querySelector('#events-table-body');

  // Timeline
  const timelineSvg = document.querySelector('#timeline-chart');
  const timelineScrubber = document.querySelector('#timeline-scrubber');
  const scrubberTimeDisplay = document.querySelector('#scrubber-time-display');

  // Replay Engine
  const liveIndicator = document.querySelector('#live-indicator');
  const liveStateText = document.querySelector('#live-state-text');
  const replayTimeReadout = document.querySelector('#replay-time-readout');
  const replayWindowReadout = document.querySelector('#replay-window-readout');
  const replayProbReadout = document.querySelector('#replay-prob-readout');
  const replayStatusBadge = document.querySelector('#replay-status-badge');
  const replayProbBar = document.querySelector('#replay-prob-bar');
  const replayPersistenceReadout = document.querySelector('#replay-persistence-readout');
  const replayPersistenceSub = document.querySelector('#replay-persistence-sub');
  const btnReplayPlay = document.querySelector('#btn-replay-play');
  const btnReplayPause = document.querySelector('#btn-replay-pause');
  const btnReplayReset = document.querySelector('#btn-replay-reset');
  const replaySpeed = document.querySelector('#replay-speed');

  // Waveform & XAI
  const channelSelector = document.querySelector('#channel-selector');
  const waveformChart = document.querySelector('#waveform-chart');
  const waveformCaption = document.querySelector('#waveform-caption');
  const attributionChart = document.querySelector('#attribution-chart');
  const xaiCaption = document.querySelector('#xai-caption');
  const importantRegionsList = document.querySelector('#important-regions-list');
  const importantChannelsList = document.querySelector('#important-channels-list');

  // Specs
  const specCheckpoint = document.querySelector('#spec-checkpoint');
  const specThreshold = document.querySelector('#spec-threshold');
  const specPersistence = document.querySelector('#spec-persistence');

  // State
  let selectedFile = null;
  let analysis = null;
  let currentWindowIndex = 0;
  let replayIntervalId = null;
  let isReplaying = false;

  // Helpers
  const formatBytes = (bytes) => {
    if (bytes < 1024 * 1024) return `${(bytes / 1024).toFixed(1)} KB`;
    return `${(bytes / (1024 * 1024)).toFixed(1)} MB`;
  };

  const formatSeconds = (sec) => {
    const s = Math.floor(sec % 60);
    const m = Math.floor((sec / 60) % 60);
    const h = Math.floor(sec / 3600);
    const pad = (n) => String(n).padStart(2, '0');
    return `${pad(h)}:${pad(m)}:${pad(s)}`;
  };

  const setStatus = (text, type = 'ready') => {
    if (statusText) statusText.textContent = text;
    if (connectionStatus) connectionStatus.className = `status-indicator ${type}`;
  };

  const showError = (message) => {
    if (errorMessage) {
      errorMessage.textContent = message;
      errorMessage.hidden = false;
    }
    if (progressBarContainer) progressBarContainer.hidden = true;
    setStatus('Analysis Error', 'error');
  };

  const clearError = () => {
    if (errorMessage) {
      errorMessage.hidden = true;
      errorMessage.textContent = '';
    }
  };

  // Upload & File Selection
  function selectFile(file) {
    clearError();
    if (resultsDashboard) resultsDashboard.hidden = true;
    pauseReplay();

    if (!file) return;

    if (!file.name.toLowerCase().endsWith('.edf')) {
      selectedFile = null;
      if (fileInfoRow) fileInfoRow.hidden = true;
      showError('Please upload an EDF recording (.edf).');
      return;
    }

    if (file.size === 0) {
      selectedFile = null;
      if (fileInfoRow) fileInfoRow.hidden = true;
      showError('The selected EDF file is empty (0 bytes).');
      return;
    }

    selectedFile = file;
    if (fileName) fileName.textContent = file.name;
    if (fileSize) fileSize.textContent = formatBytes(file.size);
    if (fileInfoRow) fileInfoRow.hidden = false;
    if (analyzeButton) analyzeButton.disabled = false;
    setStatus('Ready to Analyze', 'ready');
  }

  if (browseButton) {
    browseButton.addEventListener('click', (e) => {
      e.stopPropagation();
      if (fileInput) fileInput.click();
    });
  }

  if (dropZone) {
    dropZone.addEventListener('click', () => fileInput && fileInput.click());
    ['dragenter', 'dragover'].forEach((eventName) => {
      dropZone.addEventListener(eventName, (e) => {
        e.preventDefault();
        dropZone.classList.add('dragging');
      });
    });

    ['dragleave', 'drop'].forEach((eventName) => {
      dropZone.addEventListener(eventName, (e) => {
        e.preventDefault();
        dropZone.classList.remove('dragging');
      });
    });

    dropZone.addEventListener('drop', (e) => {
      selectFile(e.dataTransfer.files[0]);
    });
  }

  if (fileInput) {
    fileInput.addEventListener('change', () => selectFile(fileInput.files[0]));
  }

  // SVG Drawing Helpers
  function createSvgContext(svg, height = 240) {
    if (!svg) return null;
    const width = Math.max(500, svg.clientWidth || svg.parentElement?.clientWidth || 700);
    svg.innerHTML = '';
    svg.setAttribute('viewBox', `0 0 ${width} ${height}`);
    return {
      width,
      height,
      padLeft: 56,
      padRight: 20,
      padTop: 18,
      padBottom: 32,
    };
  }

  function createSvgElement(tag, attrs = {}) {
    const elem = document.createElementNS('http://www.w3.org/2000/svg', tag);
    Object.entries(attrs).forEach(([key, val]) => elem.setAttribute(key, val));
    return elem;
  }

  // Draw Multichannel Waveform
  function drawWaveform() {
    if (!analysis || !analysis.waveform || !waveformChart) return;

    const wave = analysis.waveform;
    const channelIdx = parseInt(channelSelector ? channelSelector.value : 0, 10) || 0;
    const samples = wave.samples[channelIdx];
    if (!samples || samples.length === 0) return;

    const ctx = createSvgContext(waveformChart, 260);
    if (!ctx) return;
    const plotWidth = ctx.width - ctx.padLeft - ctx.padRight;
    const plotHeight = ctx.height - ctx.padTop - ctx.padBottom;

    // Background grid
    const gridY = [0.25, 0.5, 0.75];
    gridY.forEach((ratio) => {
      const y = ctx.padTop + ratio * plotHeight;
      ctxContextLine(waveformChart, ctx.padLeft, ctx.width - ctx.padRight, y, y, '#f1f5f9', 1, '4,4');
    });

    // Zero-line
    const midY = ctx.padTop + 0.5 * plotHeight;
    ctxContextLine(waveformChart, ctx.padLeft, ctx.width - ctx.padRight, midY, midY, '#e2e8f0', 1.2);

    // Compute min / max
    let min = Infinity;
    let max = -Infinity;
    for (let i = 0; i < samples.length; i++) {
      if (samples[i] < min) min = samples[i];
      if (samples[i] > max) max = samples[i];
    }
    const spread = Math.max(0.1, max - min);

    // Signal path
    let pathD = '';
    for (let i = 0; i < samples.length; i++) {
      const x = ctx.padLeft + (i / (samples.length - 1)) * plotWidth;
      const normY = (samples[i] - min) / spread;
      const y = ctx.padTop + (1 - normY) * plotHeight;
      pathD += (i === 0 ? 'M' : 'L') + x.toFixed(1) + ',' + y.toFixed(1);
    }

    const path = createSvgElement('path', {
      d: pathD,
      fill: 'none',
      stroke: '#0284c7',
      'stroke-width': '1.5',
      'stroke-linecap': 'round',
      'stroke-linejoin': 'round',
    });
    waveformChart.appendChild(path);

    // Axes
    drawAxes(waveformChart, ctx, wave.start_seconds, wave.stop_seconds, 'Time in Recording (seconds)', `${min.toFixed(1)}`, `${max.toFixed(1)}`);

    const channelName = wave.channel_names[channelIdx] || `Ch ${channelIdx}`;
    if (waveformCaption) {
      waveformCaption.textContent = `Representative 4.0-second window (${wave.start_seconds.toFixed(1)}s – ${wave.stop_seconds.toFixed(1)}s) • Channel: ${channelName} • Normalization: Z-score CAR`;
    }
  }

  // Draw Grad-CAM Attribution Overlay
  function drawGradCamAttribution() {
    if (!analysis || !analysis.explainability || !attributionChart) return;

    const wave = analysis.waveform;
    const xai = analysis.explainability;
    const channelIdx = parseInt(channelSelector ? channelSelector.value : 0, 10) || 0;
    const samples = wave.samples[channelIdx];

    const ctx = createSvgContext(attributionChart, 300);
    if (!ctx) return;
    const plotWidth = ctx.width - ctx.padLeft - ctx.padRight;
    const plotHeight = ctx.height - ctx.padTop - ctx.padBottom;
    const midSplit = ctx.padTop + plotHeight * 0.48;

    // Highlight Important Temporal Regions with vibrant gold/yellow overlays
    if (xai.important_temporal_regions && xai.important_temporal_regions.length > 0) {
      xai.important_temporal_regions.forEach((region) => {
        const xStart = ctx.padLeft + Math.max(0, (region.start_seconds - wave.start_seconds) / 4.0) * plotWidth;
        const xEnd = ctx.padLeft + Math.min(1, (region.end_seconds - wave.start_seconds) / 4.0) * plotWidth;
        const width = Math.max(3, xEnd - xStart);

        const highlightRect = createSvgElement('rect', {
          x: xStart.toFixed(1),
          y: ctx.padTop,
          width: width.toFixed(1),
          height: plotHeight,
          fill: 'rgba(250, 204, 21, 0.40)',
          stroke: '#eab308',
          'stroke-dasharray': '3,3',
          'stroke-width': '1.5',
        });
        attributionChart.appendChild(highlightRect);
      });
    }

    // Upper curve: Waveform
    if (samples && samples.length > 0) {
      let min = Math.min(...samples);
      let max = Math.max(...samples);
      let spread = Math.max(0.1, max - min);
      let waveD = '';
      for (let i = 0; i < samples.length; i++) {
        const x = ctx.padLeft + (i / (samples.length - 1)) * plotWidth;
        const norm = (samples[i] - min) / spread;
        const y = ctx.padTop + (1 - norm) * (midSplit - ctx.padTop - 8);
        waveD += (i === 0 ? 'M' : 'L') + x.toFixed(1) + ',' + y.toFixed(1);
      }
      attributionChart.appendChild(createSvgElement('path', {
        d: waveD,
        fill: 'none',
        stroke: '#0284c7',
        'stroke-width': '1.3',
      }));

      // Label for wave
      const waveLabel = createSvgElement('text', {
        x: ctx.padLeft + 4,
        y: ctx.padTop + 14,
        fill: '#0369a1',
        'font-size': '11',
        'font-weight': '600',
      });
      waveLabel.textContent = `Raw EEG: ${wave.channel_names[channelIdx]}`;
      attributionChart.appendChild(waveLabel);
    }

    // Divider line
    ctxContextLine(attributionChart, ctx.padLeft, ctx.width - ctx.padRight, midSplit, midSplit, '#e2e8f0', 1);

    // Lower curve: Grad-CAM Temporal Importance
    const gradMap = xai.gradcam_temporal_map;
    if (gradMap && gradMap.length > 0) {
      let minG = Math.min(...gradMap);
      let maxG = Math.max(...gradMap);
      let spreadG = Math.max(1e-6, maxG - minG);
      let gradD = '';
      const topGrad = midSplit + 16;
      const gradH = ctx.height - ctx.padBottom - topGrad;

      for (let i = 0; i < gradMap.length; i++) {
        const x = ctx.padLeft + (i / (gradMap.length - 1)) * plotWidth;
        const norm = (gradMap[i] - minG) / spreadG;
        const y = topGrad + (1 - norm) * gradH;
        gradD += (i === 0 ? 'M' : 'L') + x.toFixed(1) + ',' + y.toFixed(1);
      }

      attributionChart.appendChild(createSvgElement('path', {
        d: gradD,
        fill: 'none',
        stroke: '#8b5cf6',
        'stroke-width': '1.8',
      }));

      const gradLabel = createSvgElement('text', {
        x: ctx.padLeft + 4,
        y: topGrad + 14,
        fill: '#7c3aed',
        'font-size': '11',
        'font-weight': '600',
      });
      gradLabel.textContent = 'Grad-CAM model attribution (Final Conv2D Layer)';
      attributionChart.appendChild(gradLabel);
    }

    // Time Axis
    drawAxes(attributionChart, ctx, wave.start_seconds, wave.stop_seconds, 'Time in Recording (seconds)', '0.0', 'Attribution');

    if (xaiCaption) {
      xaiCaption.textContent = `Target class: ${xai.target_class.toUpperCase()} • Representative segment: ${wave.start_seconds.toFixed(1)}s – ${wave.stop_seconds.toFixed(1)}s`;
    }
  }

  // Draw Full Recording Probability Timeline
  function drawTimelineChart() {
    if (!analysis || !analysis.segments_summary || analysis.segments_summary.length === 0 || !timelineSvg) return;

    const segments = analysis.segments_summary;
    const threshold = analysis.prediction.threshold || 0.50;
    const events = analysis.temporal_events || [];
    const totalDuration = analysis.recording.duration_seconds;

    const ctx = createSvgContext(timelineSvg, 180);
    if (!ctx) return;
    const plotWidth = ctx.width - ctx.padLeft - ctx.padRight;
    const plotHeight = ctx.height - ctx.padTop - ctx.padBottom;

    // Background horizontal gridlines (0%, 25%, 50%, 75%, 100%)
    [0.0, 0.25, 0.5, 0.75, 1.0].forEach((level) => {
      const y = ctx.padTop + (1 - level) * plotHeight;
      ctxContextLine(timelineSvg, ctx.padLeft, ctx.width - ctx.padRight, y, y, '#f1f5f9', 1);

      // Y-axis label
      const yLabel = createSvgElement('text', {
        x: ctx.padLeft - 8,
        y: y + 3,
        'text-anchor': 'end',
        fill: '#94a3b8',
        'font-size': '10',
        'font-family': 'var(--font-mono)',
      });
      yLabel.textContent = `${Math.round(level * 100)}%`;
      timelineSvg.appendChild(yLabel);
    });

    // Detected Event Shaded Bands
    events.forEach((ev) => {
      const xStart = ctx.padLeft + (ev.start_seconds / totalDuration) * plotWidth;
      const xEnd = ctx.padLeft + (ev.stop_seconds / totalDuration) * plotWidth;
      const width = Math.max(3, xEnd - xStart);

      const rect = createSvgElement('rect', {
        x: xStart.toFixed(1),
        y: ctx.padTop,
        width: width.toFixed(1),
        height: plotHeight,
        fill: 'rgba(220, 38, 38, 0.18)',
        stroke: 'rgba(220, 38, 38, 0.6)',
        'stroke-width': '1',
      });
      timelineSvg.appendChild(rect);
    });

    // Probability Area and Line
    let pathD = '';
    let areaD = `M${ctx.padLeft},${ctx.padTop + plotHeight} `;

    for (let i = 0; i < segments.length; i++) {
      const x = ctx.padLeft + (segments[i].start_seconds / totalDuration) * plotWidth;
      const prob = Math.min(1.0, Math.max(0.0, segments[i].seizure_probability));
      const y = ctx.padTop + (1 - prob) * plotHeight;

      const pt = `${x.toFixed(1)},${y.toFixed(1)}`;
      pathD += (i === 0 ? 'M' : 'L') + pt;
      areaD += 'L' + pt + ' ';
    }
    areaD += `L${ctx.width - ctx.padRight},${ctx.padTop + plotHeight} Z`;

    const areaElem = createSvgElement('path', {
      d: areaD,
      fill: 'rgba(2, 132, 199, 0.12)',
    });
    timelineSvg.appendChild(areaElem);

    const lineElem = createSvgElement('path', {
      d: pathD,
      fill: 'none',
      stroke: '#0284c7',
      'stroke-width': '1.4',
    });
    timelineSvg.appendChild(lineElem);

    // Active Scrubber Cursor Indicator
    const currentSegment = segments[currentWindowIndex] || segments[0];
    const cursorX = ctx.padLeft + (currentSegment.start_seconds / totalDuration) * plotWidth;

    const cursorLine = createSvgElement('line', {
      id: 'timeline-cursor-line',
      x1: cursorX.toFixed(1),
      x2: cursorX.toFixed(1),
      y1: ctx.padTop,
      y2: ctx.padTop + plotHeight,
      stroke: '#ea580c',
      'stroke-width': '2',
      'stroke-dasharray': '3,3',
    });
    timelineSvg.appendChild(cursorLine);

    // X-axis time markings
    const stepCount = 5;
    for (let step = 0; step <= stepCount; step++) {
      const sec = (step / stepCount) * totalDuration;
      const x = ctx.padLeft + (step / stepCount) * plotWidth;
      const tick = createSvgElement('line', {
        x1: x,
        x2: x,
        y1: ctx.padTop + plotHeight,
        y2: ctx.padTop + plotHeight + 5,
        stroke: '#94a3b8',
      });
      timelineSvg.appendChild(tick);

      const label = createSvgElement('text', {
        x,
        y: ctx.padTop + plotHeight + 18,
        'text-anchor': 'middle',
        fill: '#64748b',
        'font-size': '10',
        'font-family': 'var(--font-mono)',
      });
      label.textContent = formatSeconds(sec);
      timelineSvg.appendChild(label);
    }

    // Direct click on SVG timeline to seek
    timelineSvg.onclick = (e) => {
      const rect = timelineSvg.getBoundingClientRect();
      const clickX = e.clientX - rect.left;
      const ratio = Math.max(0, Math.min(1, (clickX - ctx.padLeft) / plotWidth));
      const targetWindow = Math.floor(ratio * (segments.length - 1));
      jumpToWindow(targetWindow);
    };
  }

  function updateTimelineCursor(index) {
    if (!analysis || !analysis.segments_summary || !timelineSvg) return;
    const totalDuration = analysis.recording.duration_seconds;
    const segments = analysis.segments_summary;
    const seg = segments[index] || segments[0];

    const ctxWidth = Math.max(500, timelineSvg.clientWidth || 700);
    const padLeft = 56;
    const padRight = 20;
    const plotWidth = ctxWidth - padLeft - padRight;

    const cursorX = padLeft + (seg.start_seconds / totalDuration) * plotWidth;
    const cursor = document.querySelector('#timeline-cursor-line');
    if (cursor) {
      cursor.setAttribute('x1', cursorX.toFixed(1));
      cursor.setAttribute('x2', cursorX.toFixed(1));
    }
  }

  function ctxContextLine(svg, x1, x2, y1, y2, stroke, width = 1, dash = '') {
    if (!svg) return;
    const line = createSvgElement('line', {
      x1, x2, y1, y2, stroke, 'stroke-width': width,
    });
    if (dash) line.setAttribute('stroke-dasharray', dash);
    svg.appendChild(line);
  }

  function drawAxes(svg, ctx, start, stop, xLabel, yMin, yMax) {
    if (!svg || !ctx) return;
    const bottomY = ctx.height - ctx.padBottom;

    // Bottom Axis Line
    ctxContextLine(svg, ctx.padLeft, ctx.width - ctx.padRight, bottomY, bottomY, '#cbd5e1', 1.2);

    // Ticks
    [0.0, 0.25, 0.5, 0.75, 1.0].forEach((ratio) => {
      const x = ctx.padLeft + ratio * (ctx.width - ctx.padLeft - ctx.padRight);
      ctxContextLine(svg, x, x, bottomY, bottomY + 5, '#94a3b8');

      const text = createSvgElement('text', {
        x,
        y: bottomY + 16,
        'text-anchor': 'middle',
        fill: '#64748b',
        'font-size': '10',
        'font-family': 'var(--font-mono)',
      });
      text.textContent = `${(start + ratio * (stop - start)).toFixed(1)}s`;
      svg.appendChild(text);
    });

    // X-axis label
    const labelElem = createSvgElement('text', {
      x: ctx.padLeft + (ctx.width - ctx.padLeft - ctx.padRight) / 2,
      y: ctx.height - 4,
      'text-anchor': 'middle',
      fill: '#475569',
      'font-size': '11',
      'font-weight': '500',
    });
    labelElem.textContent = xLabel;
    svg.appendChild(labelElem);
  }

  // Populate Dashboard Data
  function populateResults(payload) {
    analysis = payload;
    if (resultsDashboard) resultsDashboard.hidden = false;
    currentWindowIndex = 0;

    const pred = payload.prediction;
    const isSeizure = pred.is_seizure;

    // Recording / Event Result Badge
    if (predictionBadge) {
      if (isSeizure) {
        predictionBadge.textContent = 'SEIZURE EVENT DETECTED';
        predictionBadge.className = 'result-badge seizure';
        if (resultCard) resultCard.className = 'card result-card seizure-detected';
      } else {
        predictionBadge.textContent = 'NO SUSTAINED SEIZURE EVENT DETECTED';
        predictionBadge.className = 'result-badge non-seizure';
        if (resultCard) resultCard.className = 'card result-card clean-detected';
      }
    }

    // Signal Quality
    if (qualityBadge) {
      const quality = payload.quality_report;
      if (quality && quality.clean) {
        qualityBadge.textContent = 'Signal Quality: Clean';
        qualityBadge.className = 'badge badge-clean';
      } else {
        const msg = quality && quality.summary ? quality.summary : 'Artifact Warning';
        qualityBadge.textContent = `Signal Quality: ${msg}`;
        qualityBadge.className = 'badge badge-warning';
      }
    }

    // Concise Metrics (Highest Window Score, Positive Window Count, Detected Event Count)
    if (highestProbValue) highestProbValue.textContent = `${(pred.highest_single_window_probability * 100).toFixed(2)}%`;
    if (positiveWindowsValue) positiveWindowsValue.textContent = String(pred.positive_window_count);
    if (eventsCountValue) eventsCountValue.textContent = String(payload.temporal_events ? payload.temporal_events.length : 0);

    // Guarded Optional Metric Updates
    if (thresholdValBox) thresholdValBox.textContent = `${Math.round(pred.threshold * 100)}%`;
    if (persistenceValBox) persistenceValBox.textContent = `${pred.persistence_required} windows`;
    if (longestRunValue) longestRunValue.textContent = `${pred.longest_positive_run} windows`;
    if (runDurationSub) runDurationSub.textContent = `${(pred.longest_positive_run * 2.0).toFixed(1)} s duration`;
    if (positiveRatioSub) {
      const posPct = ((pred.positive_window_count / payload.recording.segment_count) * 100).toFixed(2);
      positiveRatioSub.textContent = `${posPct}% of total`;
    }
    if (meanProbValue) {
      const meanVal = pred.mean_window_score != null ? pred.mean_window_score : 0;
      meanProbValue.textContent = `${(meanVal * 100).toFixed(2)}%`;
    }
    if (thresholdText) thresholdText.textContent = pred.threshold.toFixed(2);
    if (requiredPersistenceText) requiredPersistenceText.textContent = `${pred.persistence_required} windows`;
    if (aggregationRuleText) aggregationRuleText.textContent = pred.aggregation_rule;

    // Guarded Metadata Ribbon
    if (durationVal) {
      const durSec = payload.recording.duration_seconds;
      durationVal.textContent = `${durSec.toFixed(1)} s (${(durSec / 60).toFixed(1)} min)`;
    }
    if (samplingVal) samplingVal.textContent = `${payload.recording.sampling_rate} Hz`;
    if (channelsVal) channelsVal.textContent = `${payload.recording.channels.length} Canonical`;
    if (totalWindowsVal) totalWindowsVal.textContent = `${payload.recording.segment_count} windows`;

    // Populate Channel Selector
    if (channelSelector) {
      channelSelector.innerHTML = '';
      payload.recording.channels.forEach((ch, idx) => {
        const opt = document.createElement('option');
        opt.value = String(idx);
        opt.textContent = `${idx + 1}. ${ch}`;
        channelSelector.appendChild(opt);
      });
    }

    // Populate Events Table (Strictly 6 Columns: Event #, Timeline Interval, Duration, Consecutive Windows, Peak Probability, Action)
    const events = payload.temporal_events || [];
    if (eventsTableBody && eventsSection) {
      eventsTableBody.innerHTML = '';
      if (events.length > 0) {
        eventsSection.hidden = false;
        if (eventsBadgeCount) eventsBadgeCount.textContent = `${events.length} Event${events.length > 1 ? 's' : ''}`;

        events.forEach((ev) => {
          const tr = document.createElement('tr');
          tr.innerHTML = `
            <td><strong>Event #${ev.event_id}</strong></td>
            <td><code>${formatSeconds(ev.start_seconds)} – ${formatSeconds(ev.stop_seconds)}</code></td>
            <td>${ev.duration_seconds.toFixed(1)} s</td>
            <td>${ev.window_count} windows</td>
            <td><strong style="color: var(--color-seizure);">${(ev.max_probability * 100).toFixed(1)}%</strong></td>
            <td>
              <button class="btn btn-secondary" style="padding: 4px 10px; font-size: 0.75rem;" type="button" data-window="${ev.start_window_index}">
                Inspect Segment
              </button>
            </td>
          `;
          const btn = tr.querySelector('button');
          if (btn) btn.addEventListener('click', () => jumpToWindow(ev.start_window_index));
          eventsTableBody.appendChild(tr);
        });
      } else {
        eventsSection.hidden = true;
      }
    }

    // Configure Scrubber
    if (timelineScrubber) {
      timelineScrubber.min = 0;
      timelineScrubber.max = Math.max(0, payload.recording.segment_count - 1);
      timelineScrubber.value = 0;
    }
    if (scrubberTimeDisplay) scrubberTimeDisplay.textContent = '00:00:00 (Window 0)';

    // Initial Visualizations
    drawWaveform();
    drawGradCamAttribution();
    drawTimelineChart();
    updateReplayDisplay(0, 0);

    // Scroll smoothly to results
    if (resultsDashboard) {
      resultsDashboard.scrollIntoView({ behavior: 'smooth', block: 'start' });
    }
  }

  // Jump to specific window
  window.jumpToWindow = function(idx) {
    if (!analysis || !analysis.segments_summary) return;
    const maxIdx = analysis.segments_summary.length - 1;
    currentWindowIndex = Math.max(0, Math.min(maxIdx, idx));

    if (timelineScrubber) timelineScrubber.value = currentWindowIndex;
    updateTimelineCursor(currentWindowIndex);

    // Compute cumulative consecutive run up to this window
    let run = 0;
    const threshold = analysis.prediction.threshold;
    for (let i = 0; i <= currentWindowIndex; i++) {
      if (analysis.segments_summary[i].seizure_probability >= threshold) {
        run++;
      } else {
        run = 0;
      }
    }

    updateReplayDisplay(currentWindowIndex, run);
  };

  // Replay Engine Logic
  function updateReplayDisplay(index, currentRun) {
    if (!analysis || !analysis.segments_summary) return;

    const segments = analysis.segments_summary;
    const seg = segments[index];
    if (!seg) return;

    const threshold = analysis.prediction.threshold;
    const persistenceReq = analysis.prediction.persistence_required;
    const prob = seg.seizure_probability;
    const isPos = prob >= threshold;

    // Time & Window Readout (Guarded)
    if (replayTimeReadout) replayTimeReadout.textContent = formatSeconds(seg.start_seconds);
    if (replayWindowReadout) replayWindowReadout.textContent = `Window ${index + 1} / ${segments.length}`;
    if (scrubberTimeDisplay) scrubberTimeDisplay.textContent = `${formatSeconds(seg.start_seconds)} (Window ${index + 1})`;

    // Probability & Bar
    const probPct = (prob * 100).toFixed(1);
    if (replayProbReadout) replayProbReadout.textContent = `${probPct}%`;
    if (replayProbBar) {
      replayProbBar.style.width = `${Math.min(100, prob * 100)}%`;
      replayProbBar.style.backgroundColor = isPos ? '#dc2626' : '#0284c7';
    }

    // Instantaneous Status Badge Highlight
    if (replayStatusBadge) {
      if (isPos) {
        replayStatusBadge.textContent = 'SEIZURE';
        replayStatusBadge.className = 'badge badge-alert prominent-status-badge';
      } else {
        replayStatusBadge.textContent = 'NON-SEIZURE';
        replayStatusBadge.className = 'badge badge-clean prominent-status-badge';
      }
    }

    // Persistence (Guarded)
    if (replayPersistenceReadout) replayPersistenceReadout.textContent = `${currentRun} / ${persistenceReq}`;
    const eventTriggered = currentRun >= persistenceReq;

    if (liveStateText && liveIndicator) {
      if (eventTriggered) {
        if (replayPersistenceSub) {
          replayPersistenceSub.textContent = 'SEIZURE EVENT ACTIVE';
          replayPersistenceSub.style.color = '#ef4444';
        }
        liveStateText.textContent = 'SEIZURE EVENT DETECTED';
        liveIndicator.className = 'status-dot';
        liveIndicator.style.backgroundColor = '#dc2626';
        liveIndicator.style.boxShadow = '0 0 10px #dc2626';
      } else if (isPos) {
        if (replayPersistenceSub) {
          replayPersistenceSub.textContent = 'Suspect positive spike';
          replayPersistenceSub.style.color = '#f59e0b';
        }
        liveStateText.textContent = 'SUSPECT BURST';
        liveIndicator.className = 'status-dot';
        liveIndicator.style.backgroundColor = '#f59e0b';
        liveIndicator.style.boxShadow = '0 0 8px #f59e0b';
      } else {
        if (replayPersistenceSub) {
          replayPersistenceSub.textContent = 'consecutive windows';
          replayPersistenceSub.style.color = '#64748b';
        }
        liveStateText.textContent = isReplaying ? 'REPLAY STREAMING' : 'ENGINE IDLE';
        liveIndicator.className = 'status-dot';
        liveIndicator.style.backgroundColor = isReplaying ? '#06b6d4' : '#10b981';
        liveIndicator.style.boxShadow = 'none';
      }
    }

    updateTimelineCursor(index);
  }

  function startReplay() {
    if (!analysis || !analysis.segments_summary) return;
    if (isReplaying) return;

    isReplaying = true;
    if (btnReplayPlay) btnReplayPlay.disabled = true;
    if (btnReplayPause) btnReplayPause.disabled = false;
    if (btnReplayReset) btnReplayReset.disabled = false;
    setStatus('Streaming Replay Active', 'streaming');

    const segments = analysis.segments_summary;
    const threshold = analysis.prediction.threshold;
    let run = 0;

    const speedMultiplier = parseFloat(replaySpeed ? replaySpeed.value : 5) || 5;
    // Real window stride is 2.0s = 2000ms. At 5x, each tick is 400ms.
    const intervalMs = Math.max(50, Math.round(2000 / speedMultiplier));

    replayIntervalId = setInterval(() => {
      if (currentWindowIndex >= segments.length - 1) {
        pauseReplay();
        return;
      }

      currentWindowIndex++;
      if (timelineScrubber) timelineScrubber.value = currentWindowIndex;

      if (segments[currentWindowIndex].seizure_probability >= threshold) {
        run++;
      } else {
        run = 0;
      }

      updateReplayDisplay(currentWindowIndex, run);
    }, intervalMs);
  }

  function pauseReplay() {
    if (replayIntervalId) {
      clearInterval(replayIntervalId);
      replayIntervalId = null;
    }
    isReplaying = false;
    if (btnReplayPlay) btnReplayPlay.disabled = false;
    if (btnReplayPause) btnReplayPause.disabled = true;
    setStatus('System Ready', 'ready');
    if (liveStateText) liveStateText.textContent = 'REPLAY PAUSED';
  }

  function resetReplay() {
    pauseReplay();
    currentWindowIndex = 0;
    if (timelineScrubber) timelineScrubber.value = 0;
    if (btnReplayReset) btnReplayReset.disabled = true;
    updateReplayDisplay(0, 0);
  }

  // Replay Listeners
  if (btnReplayPlay) btnReplayPlay.addEventListener('click', startReplay);
  if (btnReplayPause) btnReplayPause.addEventListener('click', pauseReplay);
  if (btnReplayReset) btnReplayReset.addEventListener('click', resetReplay);

  if (replaySpeed) {
    replaySpeed.addEventListener('change', () => {
      if (isReplaying) {
        pauseReplay();
        startReplay();
      }
    });
  }

  if (timelineScrubber) {
    timelineScrubber.addEventListener('input', (e) => {
      pauseReplay();
      const idx = parseInt(e.target.value, 10);
      jumpToWindow(idx);
    });
  }

  if (channelSelector) {
    channelSelector.addEventListener('change', () => {
      drawWaveform();
      drawGradCamAttribution();
    });
  }

  window.addEventListener('resize', () => {
    drawWaveform();
    drawGradCamAttribution();
    drawTimelineChart();
  });

  // Execute Analysis Request
  if (analyzeButton) {
    analyzeButton.addEventListener('click', async () => {
      if (!selectedFile) {
        showError('Please choose an EDF file first.');
        return;
      }

      clearError();
      analyzeButton.disabled = true;
      if (progressBarContainer) progressBarContainer.hidden = false;
      if (progressBar) progressBar.style.width = '20%';
      setStatus('Uploading recording to inference service...', 'processing');

      try {
        const formData = new FormData();
        formData.append('file', selectedFile);

        if (progressBar) progressBar.style.width = '45%';
        setStatus('Preprocessing EEG (Filtering, CAR, Z-score, Windowing)...', 'processing');

        const response = await fetch(`${apiBase}/api/eeg/analyze`, {
          method: 'POST',
          body: formData,
        });

        if (progressBar) progressBar.style.width = '85%';
        setStatus('Evaluating CNN-BiLSTM & Grad-CAM attribution...', 'processing');

        const payload = await response.json().catch(() => ({}));

        if (!response.ok) {
          throw new Error(payload.detail || `Analysis failed with HTTP status ${response.status}.`);
        }

        if (progressBar) progressBar.style.width = '100%';
        populateResults(payload);
        setStatus('Analysis Complete', 'ready');
        setTimeout(() => {
          if (progressBarContainer) progressBarContainer.hidden = true;
        }, 600);
      } catch (err) {
        showError(err.message === 'Failed to fetch'
          ? 'Could not connect to the EEG analysis backend. Please ensure the server is running.'
          : err.message);
      } finally {
        analyzeButton.disabled = false;
      }
    });
  }

})();
