"""Professional GUI for the live transcriber (PySide6).

Design language: dark glass.
  * Frameless QWidget so we can paint our own translucent background.
  * Custom titlebar with drag, pin (always-on-top), opacity slider, minimise/close.
  * Two scrollable panels: source-language transcript on top, Spanish
    translation below. Both show a "live caption" line at the bottom
    (italic, dim) that is replaced when its segment finalises.
  * Status strip with device · GPU · model · detected language · VU meter.
  * Metrics strip with segments · RTF · latency · CPU% · GPU% · VRAM.
  * Edge resize via QSizeGrip in the bottom-right corner.

Threading: the heavy pipeline runs in OrchestratorBridge (QThread), which
emits Qt signals consumed by widget slots in the main GUI thread. No direct
asyncio calls happen from widget code.
"""

from __future__ import annotations

import logging
import sys
from dataclasses import dataclass
from datetime import datetime
from html import escape

from PySide6.QtCore import (
    QPoint,
    QSize,
    Qt,
    QTimer,
    Signal,
)
from PySide6.QtGui import (
    QColor,
    QFont,
    QGuiApplication,
    QIcon,
    QMouseEvent,
    QPainter,
    QPainterPath,
    QPaintEvent,
    QTextCursor,
)
from PySide6.QtWidgets import (
    QApplication,
    QFrame,
    QHBoxLayout,
    QLabel,
    QMainWindow,
    QProgressBar,
    QPushButton,
    QSizeGrip,
    QSizePolicy,
    QSlider,
    QStackedWidget,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)

from ..hardware.monitor import HardwareMonitor
from ..pipeline.orchestrator import (
    EvtAudioLevel,
    EvtError,
    EvtLoading,
    EvtPreviewTranscript,
    EvtSystemReady,
    EvtTranscript,
    Orchestrator,
)
from .gui_bridge import OrchestratorBridge

log = logging.getLogger(__name__)


# ---------- Palette ----------

class C:
    BG_OUTER = QColor(15, 18, 28, 215)
    BG_CARD = QColor(22, 26, 39, 235)
    BG_PANEL = QColor(28, 33, 48, 200)
    BG_TITLE = QColor(15, 18, 28, 240)
    BORDER = QColor(92, 219, 255, 50)
    ACCENT = "#5cdbff"
    ACCENT_DIM = "rgba(92,219,255,0.45)"
    TEXT = "#e8edf5"
    TEXT_DIM = "#aab2c5"
    TEXT_MUTED = "#6e7689"
    SUCCESS = "#5ee8a8"
    WARN = "#f7c862"
    DANGER = "#ff5c7c"


# ---------- Stylesheet ----------

QSS = f"""
QWidget {{
    color: {C.TEXT};
    font-family: "Inter", "Segoe UI", "SF Pro Text", system-ui, sans-serif;
    font-size: 11pt;
}}

#TitleBar {{
    background: transparent;
}}
#TitleBar QLabel#AppTitle {{
    color: {C.TEXT};
    font-weight: 600;
    font-size: 11pt;
    padding-left: 8px;
}}
#TitleBar QLabel#AppSubtitle {{
    color: {C.TEXT_MUTED};
    font-size: 9pt;
    padding-left: 8px;
}}
#TitleBar QPushButton {{
    background: transparent;
    color: {C.TEXT_DIM};
    border: none;
    padding: 4px 10px;
    border-radius: 6px;
    font-size: 10pt;
}}
#TitleBar QPushButton:hover {{
    background: rgba(255,255,255,0.06);
    color: {C.TEXT};
}}
#TitleBar QPushButton#CloseBtn:hover {{
    background: rgba(255,92,124,0.20);
    color: #ff8da3;
}}
#TitleBar QPushButton#PinBtn[on="true"] {{
    color: {C.ACCENT};
}}

#StatusStrip {{
    background: transparent;
    color: {C.TEXT_DIM};
    font-size: 9pt;
}}
#StatusStrip QLabel {{
    color: {C.TEXT_DIM};
    padding: 0 4px;
}}
#StatusStrip QLabel[role="key"] {{
    color: {C.TEXT_MUTED};
    font-size: 8pt;
    text-transform: uppercase;
    letter-spacing: 1px;
}}
#StatusStrip QLabel[role="value"] {{
    color: {C.TEXT};
    font-weight: 600;
}}
#StatusStrip QLabel[role="lang"] {{
    color: {C.ACCENT};
    font-weight: 700;
}}

#PanelHeader {{
    color: {C.TEXT_MUTED};
    font-size: 8pt;
    font-weight: 600;
    letter-spacing: 2px;
    padding: 6px 2px 2px 2px;
}}
#PanelHeaderTag {{
    color: {C.ACCENT};
    font-weight: 700;
    letter-spacing: 1px;
}}

QTextEdit#TranscriptText, QTextEdit#TranslationText {{
    background: transparent;
    border: 1px solid rgba(92,219,255,0.10);
    border-radius: 10px;
    padding: 10px 12px;
    color: {C.TEXT};
    selection-background-color: rgba(92,219,255,0.35);
    selection-color: white;
}}

#MetricsStrip {{
    background: transparent;
    color: {C.TEXT_MUTED};
    font-size: 9pt;
}}
#MetricsStrip QLabel[role="key"] {{
    color: {C.TEXT_MUTED};
    font-size: 8pt;
    letter-spacing: 1px;
}}
#MetricsStrip QLabel[role="value"] {{
    color: {C.TEXT};
    font-weight: 600;
}}
#MetricsStrip QLabel[role="good"] {{
    color: {C.SUCCESS};
    font-weight: 600;
}}
#MetricsStrip QLabel[role="warn"] {{
    color: {C.WARN};
    font-weight: 600;
}}

QSlider::groove:horizontal {{
    height: 4px;
    background: rgba(255,255,255,0.08);
    border-radius: 2px;
}}
QSlider::handle:horizontal {{
    width: 12px;
    height: 12px;
    margin: -5px 0;
    background: {C.ACCENT};
    border-radius: 6px;
}}
QSlider::sub-page:horizontal {{
    background: {C.ACCENT_DIM};
    border-radius: 2px;
}}

QSizeGrip {{
    image: none;
    background: transparent;
    width: 14px;
    height: 14px;
}}

#LoadingScreen QLabel#LoadTitle {{
    color: {C.TEXT};
    font-size: 18pt;
    font-weight: 600;
}}
#LoadingScreen QLabel#LoadStep {{
    color: {C.ACCENT};
    font-size: 11pt;
    font-weight: 600;
}}
#LoadingScreen QLabel#LoadHint {{
    color: {C.TEXT_MUTED};
    font-size: 9pt;
}}
QProgressBar {{
    background: rgba(255,255,255,0.06);
    border: none;
    border-radius: 4px;
    height: 8px;
    text-align: center;
    color: transparent;
}}
QProgressBar::chunk {{
    background-color: {C.ACCENT};
    border-radius: 4px;
}}
"""


# ---------- Custom widgets ----------

class TitleBar(QWidget):
    minimise_requested = Signal()
    close_requested = Signal()
    pin_toggled = Signal(bool)
    opacity_changed = Signal(float)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setObjectName("TitleBar")
        self.setFixedHeight(44)
        self._drag_pos: QPoint | None = None

        title = QLabel("Realtime Transcriber")
        title.setObjectName("AppTitle")
        self._subtitle = QLabel("inicializando…")
        self._subtitle.setObjectName("AppSubtitle")

        # Opacity slider
        self._opacity = QSlider(Qt.Horizontal)
        self._opacity.setRange(60, 100)
        self._opacity.setValue(94)
        self._opacity.setFixedWidth(110)
        self._opacity.setToolTip("Opacidad")
        self._opacity.valueChanged.connect(
            lambda v: self.opacity_changed.emit(v / 100.0)
        )

        self._pin = QPushButton("📌")
        self._pin.setToolTip("Mantener siempre encima")
        self._pin.setProperty("on", "true")
        self._pin.setCheckable(True)
        self._pin.setChecked(True)
        self._pin.clicked.connect(self._on_pin_clicked)

        self._min = QPushButton("—")
        self._min.setToolTip("Minimizar")
        self._min.clicked.connect(self.minimise_requested)

        self._close = QPushButton("✕")
        self._close.setObjectName("CloseBtn")
        self._close.setToolTip("Cerrar")
        self._close.clicked.connect(self.close_requested)

        h = QHBoxLayout(self)
        h.setContentsMargins(12, 4, 8, 4)
        h.setSpacing(4)
        h.addWidget(title)
        h.addWidget(self._subtitle, 1)
        h.addWidget(QLabel("Opacidad"))
        h.addWidget(self._opacity)
        h.addSpacing(6)
        h.addWidget(self._pin)
        h.addWidget(self._min)
        h.addWidget(self._close)

    def set_subtitle(self, s: str) -> None:
        self._subtitle.setText(s)

    def _on_pin_clicked(self) -> None:
        on = self._pin.isChecked()
        self._pin.setProperty("on", "true" if on else "false")
        self._pin.style().unpolish(self._pin)
        self._pin.style().polish(self._pin)
        self.pin_toggled.emit(on)

    # Drag support
    def mousePressEvent(self, e: QMouseEvent) -> None:
        if e.button() == Qt.LeftButton:
            self._drag_pos = e.globalPosition().toPoint() - self.window().frameGeometry().topLeft()
            e.accept()

    def mouseMoveEvent(self, e: QMouseEvent) -> None:
        if e.buttons() & Qt.LeftButton and self._drag_pos is not None:
            self.window().move(e.globalPosition().toPoint() - self._drag_pos)
            e.accept()

    def mouseReleaseEvent(self, e: QMouseEvent) -> None:
        self._drag_pos = None


class StatusStrip(QWidget):
    def __init__(self):
        super().__init__()
        self.setObjectName("StatusStrip")

        def chip(key: str, val_role: str = "value") -> tuple[QLabel, QLabel]:
            k = QLabel(key)
            k.setProperty("role", "key")
            v = QLabel("—")
            v.setProperty("role", val_role)
            return k, v

        self._device_k, self._device_v = chip("Dispositivo")
        self._gpu_k, self._gpu_v = chip("GPU")
        self._model_k, self._model_v = chip("Modelo")
        self._lang_k, self._lang_v = chip("Idioma", val_role="lang")

        self._level = LevelBar()

        row = QHBoxLayout(self)
        row.setContentsMargins(14, 6, 14, 6)
        row.setSpacing(8)
        for k, v in (
            (self._device_k, self._device_v),
            (self._gpu_k, self._gpu_v),
            (self._model_k, self._model_v),
            (self._lang_k, self._lang_v),
        ):
            row.addWidget(k)
            row.addWidget(v)
            row.addSpacing(12)
        row.addStretch(1)
        row.addWidget(self._level)

    def set_device(self, name: str, api: str) -> None:
        self._device_v.setText(f"{name}  · {api}")

    def set_gpu(self, name: str) -> None:
        self._gpu_v.setText(name)

    def set_model(self, name: str, compute: str) -> None:
        self._model_v.setText(f"{name}  · {compute}")

    def set_language(self, lang: str, prob: float) -> None:
        if not lang:
            self._lang_v.setText("detectando…")
        else:
            self._lang_v.setText(f"{lang.upper()} · {prob:.2f}")

    def set_level(self, rms: float, peak: float) -> None:
        self._level.set_level(rms, peak)


class LevelBar(QWidget):
    """Compact VU meter."""

    def __init__(self):
        super().__init__()
        self.setFixedSize(QSize(170, 12))
        self._rms = 0.0
        self._peak = 0.0

    def set_level(self, rms: float, peak: float) -> None:
        self._rms = rms
        self._peak = peak
        self.update()

    def paintEvent(self, _e: QPaintEvent) -> None:
        p = QPainter(self)
        p.setRenderHint(QPainter.Antialiasing)
        w, h = self.width(), self.height()
        # Background
        path = QPainterPath()
        path.addRoundedRect(0, 0, w, h, 4, 4)
        p.fillPath(path, QColor(255, 255, 255, 18))
        # RMS bar (scaled)
        bar_frac = max(0.0, min(1.0, self._rms * 6))
        bar_w = int(w * bar_frac)
        if bar_w > 0:
            grad = QColor(92, 219, 255, 220)
            if bar_frac > 0.85:
                grad = QColor(255, 92, 124, 220)
            elif bar_frac > 0.65:
                grad = QColor(247, 200, 98, 220)
            p.fillRect(0, 0, bar_w, h, grad)
        # Peak marker
        peak_frac = max(0.0, min(1.0, self._peak))
        x = int(w * peak_frac)
        if x > 0:
            p.fillRect(x - 1, 0, 2, h, QColor(255, 255, 255, 200))
        p.end()


@dataclass
class PreviewState:
    segment_id: int | None = None
    block_position: int | None = None  # cursor position where preview starts


class TranscriptPanel(QTextEdit):
    """A scrolling text panel supporting finalized blocks plus a live preview
    block at the end which is replaced on every update."""

    def __init__(self, role_tag: str, accent_color: str, placeholder: str,
                 object_name: str = "TranscriptText"):
        super().__init__()
        self.setObjectName(object_name)
        self.setReadOnly(True)
        self.setUndoRedoEnabled(False)
        self.setFrameStyle(QFrame.NoFrame)
        self.setVerticalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        self.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        # Make sure the QTextEdit viewport stays transparent so the painted
        # glass background underneath shows through.
        self.viewport().setAutoFillBackground(False)
        self.setStyleSheet(
            "QTextEdit { background: rgba(28,33,48,140); border: 1px solid rgba(92,219,255,0.18); "
            "border-radius: 10px; padding: 10px 12px; color: " + C.TEXT + "; }"
        )
        self._role_tag = role_tag
        self._accent = accent_color
        self._preview = PreviewState()
        self._placeholder = placeholder
        self._show_placeholder()

    def _show_placeholder(self) -> None:
        html = (
            f"<p style='color:{C.TEXT_MUTED};font-style:italic;margin:6px 0'>"
            f"{escape(self._placeholder)}</p>"
        )
        self.setHtml(html)

    def _scroll_to_bottom(self) -> None:
        sb = self.verticalScrollBar()
        sb.setValue(sb.maximum())

    def append_final(self, ts: str, lang: str, text: str) -> None:
        if self._preview.segment_id is not None:
            self._remove_preview()
        # If placeholder is showing, clear it first.
        if self.toPlainText().strip() == self._placeholder:
            self.clear()

        ts_html = f"<span style='color:{C.TEXT_MUTED};font-size:9pt'>{escape(ts)}</span>"
        tag_html = (
            f"<span style='color:{self._accent};font-weight:700;font-size:9pt;"
            f"background:rgba(92,219,255,0.10);padding:1px 6px;"
            f"border-radius:4px;margin-right:6px'>{escape(lang.upper())}</span>"
        )
        body_html = f"<span style='color:{C.TEXT}'>{escape(text)}</span>"
        cursor = self.textCursor()
        cursor.movePosition(QTextCursor.End)
        cursor.insertHtml(
            f"<p style='margin:8px 0;line-height:1.45'>{ts_html} {tag_html} {body_html}</p>"
        )
        self._scroll_to_bottom()

    def update_preview(self, seg_id: int, lang: str, text: str) -> None:
        if self.toPlainText().strip() == self._placeholder:
            self.clear()
        if self._preview.segment_id is not None and self._preview.segment_id != seg_id:
            # Different segment opened without us getting a final — clean up old.
            self._remove_preview()

        cursor = self.textCursor()
        if self._preview.segment_id == seg_id and self._preview.block_position is not None:
            # Replace existing preview block.
            cursor.setPosition(self._preview.block_position)
            cursor.movePosition(QTextCursor.End, QTextCursor.KeepAnchor)
            cursor.removeSelectedText()
        else:
            cursor.movePosition(QTextCursor.End)
            self._preview.block_position = cursor.position()
            self._preview.segment_id = seg_id

        tag_html = (
            f"<span style='color:{C.TEXT_MUTED};font-style:italic;font-size:9pt'>"
            f"⏵ EN VIVO · {escape(lang.upper())}</span>"
        )
        body_html = f"<span style='color:{C.TEXT_DIM};font-style:italic'>{escape(text)}</span>"
        cursor.insertHtml(
            f"<p style='margin:6px 0;line-height:1.45'>{tag_html}<br>{body_html}</p>"
        )
        self._scroll_to_bottom()

    def clear_preview(self, seg_id: int) -> None:
        if self._preview.segment_id == seg_id:
            self._remove_preview()

    def _remove_preview(self) -> None:
        if self._preview.block_position is None:
            self._preview = PreviewState()
            return
        cursor = self.textCursor()
        cursor.setPosition(self._preview.block_position)
        cursor.movePosition(QTextCursor.End, QTextCursor.KeepAnchor)
        cursor.removeSelectedText()
        # Trim trailing newline if any.
        cursor.movePosition(QTextCursor.End)
        self._preview = PreviewState()

    def clear_all(self) -> None:
        self.clear()
        self._preview = PreviewState()
        self._show_placeholder()


class MetricsStrip(QWidget):
    def __init__(self):
        super().__init__()
        self.setObjectName("MetricsStrip")
        self._n_segs = 0
        self._n_prev = 0
        self._total_asr = 0.0
        self._total_audio = 0.0
        self._last_rtf = 0.0
        self._last_latency = 0.0
        self._cpu = 0.0
        self._gpu = None
        self._vram_used = None
        self._vram_total = None

        def chip(key: str) -> tuple[QLabel, QLabel]:
            k = QLabel(key)
            k.setProperty("role", "key")
            v = QLabel("0")
            v.setProperty("role", "value")
            return k, v

        self._seg_k, self._seg_v = chip("SEGS")
        self._prev_k, self._prev_v = chip("PREVS")
        self._rtf_k, self._rtf_v = chip("RTF")
        self._lat_k, self._lat_v = chip("E2E")
        self._cpu_k, self._cpu_v = chip("CPU")
        self._gpu_k, self._gpu_v = chip("GPU")
        self._vram_k, self._vram_v = chip("VRAM")

        # Right-side buttons
        self._clear_btn = QPushButton("🗑")
        self._clear_btn.setToolTip("Limpiar")
        self._clear_btn.setCursor(Qt.PointingHandCursor)
        self._clear_btn.setStyleSheet(
            "QPushButton{background:transparent;color:" + C.TEXT_DIM + ";"
            "border:none;padding:4px 8px;border-radius:6px;}"
            "QPushButton:hover{background:rgba(255,255,255,0.08);color:" + C.TEXT + ";}"
        )

        row = QHBoxLayout(self)
        row.setContentsMargins(14, 4, 4, 4)
        row.setSpacing(6)
        for k, v in (
            (self._seg_k, self._seg_v),
            (self._prev_k, self._prev_v),
            (self._rtf_k, self._rtf_v),
            (self._lat_k, self._lat_v),
            (self._cpu_k, self._cpu_v),
            (self._gpu_k, self._gpu_v),
            (self._vram_k, self._vram_v),
        ):
            row.addWidget(k)
            row.addWidget(v)
            row.addSpacing(10)
        row.addStretch(1)
        row.addWidget(self._clear_btn)

    @property
    def clear_button(self) -> QPushButton:
        return self._clear_btn

    def on_transcript(self, tr_seconds: float, audio_seconds: float,
                       rtf: float, latency: float) -> None:
        self._n_segs += 1
        self._total_asr += tr_seconds
        self._total_audio += audio_seconds
        self._last_rtf = rtf
        self._last_latency = latency
        self._refresh()

    def on_preview(self) -> None:
        self._n_prev += 1
        self._prev_v.setText(str(self._n_prev))

    def set_hardware(self, cpu: float, gpu: float | None,
                      vram_used: float | None, vram_total: float | None) -> None:
        self._cpu = cpu
        self._gpu = gpu
        self._vram_used = vram_used
        self._vram_total = vram_total
        self._refresh()

    def reset_counters(self) -> None:
        self._n_segs = 0
        self._n_prev = 0
        self._total_asr = 0.0
        self._total_audio = 0.0
        self._last_rtf = 0.0
        self._last_latency = 0.0
        self._refresh()

    def _refresh(self) -> None:
        avg_rtf = (self._total_asr / self._total_audio) if self._total_audio else 0.0
        self._seg_v.setText(str(self._n_segs))
        self._prev_v.setText(str(self._n_prev))
        self._rtf_v.setText(f"{avg_rtf:.2f}")
        # color rtf
        self._rtf_v.setProperty("role", "good" if avg_rtf < 0.4 else "warn" if avg_rtf < 1.0 else "value")
        self._lat_v.setText(f"{self._last_latency:.1f}s")
        self._lat_v.setProperty("role", "good" if self._last_latency < 6 else "warn" if self._last_latency < 12 else "value")
        self._cpu_v.setText(f"{self._cpu:.0f}%")
        if self._gpu is not None:
            self._gpu_v.setText(f"{self._gpu:.0f}%")
        else:
            self._gpu_v.setText("—")
        if self._vram_used is not None and self._vram_total:
            self._vram_v.setText(f"{self._vram_used/1024:.1f}/{self._vram_total/1024:.1f} GB")
        else:
            self._vram_v.setText("—")
        # Force restyle after property changes.
        for lab in (self._rtf_v, self._lat_v):
            lab.style().unpolish(lab)
            lab.style().polish(lab)


class LoadingScreen(QWidget):
    """Centered loading card with progress bar + per-stage message."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setObjectName("LoadingScreen")
        self.setAttribute(Qt.WA_TranslucentBackground)

        title = QLabel("Preparando el transcriptor")
        title.setObjectName("LoadTitle")
        title.setAlignment(Qt.AlignCenter)

        self._step = QLabel("Iniciando…")
        self._step.setObjectName("LoadStep")
        self._step.setAlignment(Qt.AlignCenter)

        self._bar = QProgressBar()
        self._bar.setRange(0, 100)
        self._bar.setValue(2)
        self._bar.setFixedHeight(8)
        self._bar.setTextVisible(False)

        hint = QLabel(
            "La primera ejecución puede tardar más mientras se descargan los modelos.\n"
            "Whisper large-v3 (~3 GB) y NLLB-200 1.3B (~5 GB) quedan en caché para la próxima vez."
        )
        hint.setObjectName("LoadHint")
        hint.setAlignment(Qt.AlignCenter)
        hint.setWordWrap(True)

        inner = QVBoxLayout()
        inner.setSpacing(14)
        inner.addStretch(1)
        inner.addWidget(title)
        inner.addWidget(self._step)
        inner.addWidget(self._bar)
        inner.addSpacing(10)
        inner.addWidget(hint)
        inner.addStretch(2)

        # Constrain width of the inner column so text doesn't reflow oddly.
        card = QWidget()
        card.setLayout(inner)
        card.setMaximumWidth(520)

        wrap = QHBoxLayout(self)
        wrap.setContentsMargins(40, 40, 40, 40)
        wrap.addStretch(1)
        wrap.addWidget(card)
        wrap.addStretch(1)

    def set_loading(self, stage: str, message: str, progress: float) -> None:
        self._step.setText(message)
        self._bar.setValue(int(max(0.0, min(1.0, progress)) * 100))


# ---------- Main window ----------

class GlassCentral(QWidget):
    """Custom central widget that paints the glass background AND handles
    edge-resize of the parent window.

    Has to be on the central widget (not the QMainWindow itself) because
    QMainWindow always draws the central widget on top of its own paintEvent,
    which would otherwise mask the glass effect and we'd see a solid black
    rectangle on Windows (translucent areas with no paint show as black).

    Mouse events are routed here too — the QMainWindow never receives them
    when a central widget covers the whole client area.
    """

    RESIZE_MARGIN = 6   # px

    _LEFT, _RIGHT, _TOP, _BOTTOM = 1, 2, 4, 8

    def __init__(self, opacity_ref, parent=None):
        super().__init__(parent)
        self.setAttribute(Qt.WA_TranslucentBackground)
        self.setAttribute(Qt.WA_NoSystemBackground)
        self.setAutoFillBackground(False)
        self.setMouseTracking(True)
        self._opacity_ref = opacity_ref  # callable -> float
        self._resize_edge: int = 0
        self._resize_start_geom = None
        self._resize_start_global = None

    def paintEvent(self, _e: QPaintEvent) -> None:
        opacity = self._opacity_ref()
        p = QPainter(self)
        p.setRenderHint(QPainter.Antialiasing)
        rect = self.rect().adjusted(8, 8, -8, -8)
        path = QPainterPath()
        path.addRoundedRect(rect, 14, 14)
        outer = QColor(C.BG_OUTER)
        outer.setAlpha(int(opacity * outer.alpha()))
        p.fillPath(path, outer)
        inner_rect = rect.adjusted(2, 2, -2, -2)
        inner_path = QPainterPath()
        inner_path.addRoundedRect(inner_rect, 12, 12)
        card = QColor(C.BG_CARD)
        card.setAlpha(int(opacity * card.alpha()))
        p.fillPath(inner_path, card)
        pen = p.pen()
        pen.setColor(C.BORDER)
        pen.setWidthF(1.0)
        p.setPen(pen)
        p.drawPath(inner_path)
        p.end()

    # --- edge resize (routes to the window)
    def _edge_at(self, pos: QPoint) -> int:
        m = self.RESIZE_MARGIN
        edges = 0
        if pos.x() <= m:
            edges |= self._LEFT
        elif pos.x() >= self.width() - m:
            edges |= self._RIGHT
        if pos.y() <= m:
            edges |= self._TOP
        elif pos.y() >= self.height() - m:
            edges |= self._BOTTOM
        return edges

    def _cursor_for(self, edges: int) -> Qt.CursorShape:
        if edges == (self._LEFT | self._TOP) or edges == (self._RIGHT | self._BOTTOM):
            return Qt.SizeFDiagCursor
        if edges == (self._RIGHT | self._TOP) or edges == (self._LEFT | self._BOTTOM):
            return Qt.SizeBDiagCursor
        if edges & (self._LEFT | self._RIGHT):
            return Qt.SizeHorCursor
        if edges & (self._TOP | self._BOTTOM):
            return Qt.SizeVerCursor
        return Qt.ArrowCursor

    def mouseMoveEvent(self, e: QMouseEvent) -> None:
        if self._resize_edge and (e.buttons() & Qt.LeftButton):
            self._perform_resize(e.globalPosition().toPoint())
            e.accept()
            return
        edges = self._edge_at(e.position().toPoint())
        self.setCursor(self._cursor_for(edges))
        super().mouseMoveEvent(e)

    def mousePressEvent(self, e: QMouseEvent) -> None:
        if e.button() == Qt.LeftButton:
            edges = self._edge_at(e.position().toPoint())
            if edges:
                self._resize_edge = edges
                self._resize_start_geom = self.window().geometry()
                self._resize_start_global = e.globalPosition().toPoint()
                e.accept()
                return
        super().mousePressEvent(e)

    def mouseReleaseEvent(self, e: QMouseEvent) -> None:
        if self._resize_edge:
            self._resize_edge = 0
            self._resize_start_geom = None
            self._resize_start_global = None
            self.setCursor(Qt.ArrowCursor)
            e.accept()
            return
        super().mouseReleaseEvent(e)

    def _perform_resize(self, global_pos: QPoint) -> None:
        if not self._resize_start_geom or not self._resize_start_global:
            return
        dx = global_pos.x() - self._resize_start_global.x()
        dy = global_pos.y() - self._resize_start_global.y()
        g = self._resize_start_geom
        x, y, w, h = g.x(), g.y(), g.width(), g.height()
        win = self.window()
        min_w, min_h = win.minimumWidth(), win.minimumHeight()
        edges = self._resize_edge
        if edges & self._LEFT:
            nw = max(min_w, w - dx)
            x += (w - nw)
            w = nw
        if edges & self._RIGHT:
            w = max(min_w, w + dx)
        if edges & self._TOP:
            nh = max(min_h, h - dy)
            y += (h - nh)
            h = nh
        if edges & self._BOTTOM:
            h = max(min_h, h + dy)
        win.setGeometry(x, y, w, h)


class GlassWindow(QMainWindow):
    """Frameless, translucent main window with a custom-painted central card.

    Edge-aware resize and drag are handled in `GlassCentral` and `TitleBar`
    respectively, because the central widget covers the whole client area
    and intercepts all mouse events before they reach the QMainWindow.
    """

    def __init__(self, orchestrator: Orchestrator, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Realtime Transcriber")
        self.setWindowFlags(
            Qt.Window
            | Qt.FramelessWindowHint
            | Qt.WindowStaysOnTopHint
            | Qt.WindowMinMaxButtonsHint
        )
        self.setAttribute(Qt.WA_TranslucentBackground)
        self.setAttribute(Qt.WA_NoSystemBackground)
        self.setAutoFillBackground(False)
        self.setMinimumSize(620, 480)
        self.resize(820, 720)
        self.setStyleSheet(QSS)
        self._opacity = 0.94

        # Custom central widget that paints the glass card itself. Crucial:
        # QMainWindow does NOT honour paintEvent overrides reliably with a
        # central widget on top, so painting must live on the central widget.
        self._central = GlassCentral(opacity_ref=lambda: self._opacity, parent=self)
        self._central.setMouseTracking(True)
        self.setCentralWidget(self._central)

        # Single layout on the central widget; no inner frame to avoid extra
        # opaque layers. Children sit directly on top of the painted glass.
        v = QVBoxLayout(self._central)
        v.setContentsMargins(14, 14, 14, 14)
        v.setSpacing(0)

        # --- Title bar
        self._title = TitleBar()
        self._title.minimise_requested.connect(self.showMinimized)
        self._title.close_requested.connect(self.close)
        self._title.pin_toggled.connect(self._set_pinned)
        self._title.opacity_changed.connect(self._set_opacity)
        v.addWidget(self._title)

        # --- Stack: loading screen vs. live app. We start on loading and
        # switch to the live app only when EvtSystemReady arrives.
        self._stack = QStackedWidget()
        v.addWidget(self._stack, 1)

        self._loading = LoadingScreen()
        self._stack.addWidget(self._loading)        # index 0

        live = QWidget()
        live.setAttribute(Qt.WA_TranslucentBackground)
        live_v = QVBoxLayout(live)
        live_v.setContentsMargins(0, 0, 0, 0)
        live_v.setSpacing(0)

        self._status = StatusStrip()
        live_v.addWidget(self._status)

        body = QVBoxLayout()
        body.setContentsMargins(14, 4, 14, 4)
        body.setSpacing(8)

        body.addWidget(self._make_panel_header("ORIGINAL", "Transcripción del audio"))
        self._src_panel = TranscriptPanel(
            role_tag="ORIGINAL", accent_color=C.ACCENT,
            placeholder="Esperando audio… reproduce algo para empezar.",
            object_name="TranscriptText",
        )
        body.addWidget(self._src_panel, 1)

        body.addWidget(self._make_panel_header("ESPAÑOL", "Traducción automática"))
        self._tgt_panel = TranscriptPanel(
            role_tag="ESPAÑOL", accent_color=C.SUCCESS,
            placeholder="La traducción aparecerá aquí cuando el idioma no sea español.",
            object_name="TranslationText",
        )
        body.addWidget(self._tgt_panel, 1)

        live_v.addLayout(body, 1)

        bottom_row = QHBoxLayout()
        bottom_row.setContentsMargins(0, 0, 0, 0)
        bottom_row.setSpacing(0)
        self._metrics = MetricsStrip()
        bottom_row.addWidget(self._metrics, 1)
        grip = QSizeGrip(self)
        bottom_row.addWidget(grip, 0, Qt.AlignBottom | Qt.AlignRight)
        live_v.addLayout(bottom_row)

        self._stack.addWidget(live)                  # index 1
        self._stack.setCurrentIndex(0)               # start on loading

        # --- Wire up orchestrator
        self._orch = orchestrator
        self._bridge = OrchestratorBridge(orchestrator, parent=self)
        self._bridge.loading.connect(self._on_loading)
        self._bridge.system_ready.connect(self._on_system_ready)
        self._bridge.audio_level.connect(self._on_audio_level)
        self._bridge.preview.connect(self._on_preview)
        self._bridge.transcript.connect(self._on_transcript)
        self._bridge.error.connect(self._on_error)
        self._bridge.stopped.connect(self._on_stopped)

        self._metrics.clear_button.clicked.connect(self._on_clear)

        # --- Hardware monitor: sample every 1s, push to metrics strip.
        self._hw = HardwareMonitor()
        self._hw_timer = QTimer(self)
        self._hw_timer.setInterval(1000)
        self._hw_timer.timeout.connect(self._sample_hardware)
        self._hw_timer.start()

        self._bridge.start()

    # --- factory helpers
    def _make_panel_header(self, tag: str, subtitle: str) -> QWidget:
        w = QWidget()
        h = QHBoxLayout(w)
        h.setContentsMargins(2, 8, 2, 0)
        h.setSpacing(6)
        tag_lbl = QLabel(tag)
        tag_lbl.setObjectName("PanelHeaderTag")
        sub_lbl = QLabel(subtitle)
        sub_lbl.setObjectName("PanelHeader")
        h.addWidget(tag_lbl)
        h.addWidget(sub_lbl)
        h.addStretch(1)
        return w

    # Painting handled by GlassCentral; we just trigger repaint when opacity changes.

    # --- behaviour
    def _set_pinned(self, on: bool) -> None:
        flags = self.windowFlags()
        if on:
            flags |= Qt.WindowStaysOnTopHint
        else:
            flags &= ~Qt.WindowStaysOnTopHint
        self.setWindowFlags(flags)
        self.show()

    def _set_opacity(self, opacity: float) -> None:
        self._opacity = max(0.5, min(1.0, opacity))
        self._central.update()

    def _on_clear(self) -> None:
        self._src_panel.clear_all()
        self._tgt_panel.clear_all()
        self._metrics.reset_counters()

    # --- event slots
    def _on_loading(self, ev: EvtLoading) -> None:
        self._loading.set_loading(ev.stage, ev.message, ev.progress)
        self._title.set_subtitle(ev.message)

    def _on_system_ready(self, ev: EvtSystemReady) -> None:
        self._status.set_device(ev.device.name, ev.device.api_name)
        self._status.set_gpu(ev.accel.gpu_name or "CPU")
        self._status.set_model(
            ev.whisper_model,
            f"{ev.accel.ct2_device}/{ev.accel.ct2_compute_type}",
        )
        self._title.set_subtitle(f"escuchando {ev.device.name}")
        # Swap from loading screen to the live app.
        self._stack.setCurrentIndex(1)

    def _on_audio_level(self, ev: EvtAudioLevel) -> None:
        self._status.set_level(ev.rms, ev.peak)

    def _on_preview(self, ev: EvtPreviewTranscript) -> None:
        if not ev.text:
            return
        self._metrics.on_preview()
        self._status.set_language(ev.language, ev.language_prob)
        self._src_panel.update_preview(ev.segment_id, ev.language, ev.text)
        if ev.translation:
            self._tgt_panel.update_preview(ev.segment_id, "es", ev.translation)

    def _on_transcript(self, ev: EvtTranscript) -> None:
        tr = ev.transcript
        # Clear any preview for this segment first.
        self._src_panel.clear_preview(ev.segment_id)
        self._tgt_panel.clear_preview(ev.segment_id)
        if not tr.text:
            return
        ts = datetime.now().strftime("%H:%M:%S")
        self._src_panel.append_final(ts, tr.language, tr.text)
        if ev.translation:
            self._tgt_panel.append_final(ts, "es", ev.translation)
        self._status.set_language(tr.language, tr.language_prob)
        self._metrics.on_transcript(
            tr_seconds=tr.asr_seconds,
            audio_seconds=tr.duration_s,
            rtf=tr.rtf,
            latency=ev.end_to_end_latency_s,
        )

    def _on_error(self, ev: EvtError) -> None:
        log.error("Pipeline error: %s", ev.message)
        # Show as a final entry in the source panel so the user notices.
        self._src_panel.append_final(
            datetime.now().strftime("%H:%M:%S"),
            "ERR", f"⚠ {ev.message}",
        )

    def _on_stopped(self) -> None:
        log.info("Bridge stopped cleanly.")

    def _sample_hardware(self) -> None:
        s = self._hw.sample()
        self._metrics.set_hardware(s.cpu_percent, s.gpu_percent,
                                     s.vram_used_mb, s.vram_total_mb)

    # Resize handled by GlassCentral (the child receives mouse events).

    # --- close
    def closeEvent(self, e) -> None:
        try:
            self._hw_timer.stop()
            self._hw.shutdown()
            self._bridge.request_stop()
        except Exception:
            pass
        super().closeEvent(e)


# ---------- App entry ----------

def run_gui(orchestrator: Orchestrator) -> int:
    app = QApplication.instance() or QApplication(sys.argv)
    QGuiApplication.setApplicationDisplayName("Realtime Transcriber")
    # Force "Fusion" so the dark theme renders consistently across platforms.
    try:
        app.setStyle("Fusion")
    except Exception:
        pass

    win = GlassWindow(orchestrator)
    win.show()
    return int(app.exec())
