# Realtime System-Audio Transcriber + Translator

Captura el audio que reproduce tu sistema (podcasts, vídeos, llamadas, streams)
y lo transcribe en tiempo real. Si el idioma detectado no es español, lo
traduce automáticamente a español. Pensado para precisión por delante de
consumo de recursos, con baja latencia gracias a transcripción de vista previa.

## Stack tecnológico

| Capa | Tecnología | Por qué |
|---|---|---|
| Captura Windows | PyAudioWPatch (WASAPI loopback) | Loopback nativo sin cables virtuales |
| Captura Linux | sounddevice + PulseAudio/PipeWire monitor | Estándar en escritorios Linux modernos |
| Captura macOS | sounddevice + BlackHole | CoreAudio no expone loopback nativo |
| VAD | Silero VAD v5 (paquete oficial) | Mejor precisión que WebRTC en habla con música/ruido |
| ASR | faster-whisper large-v3 (CTranslate2) | 4-5× más rápido que openai-whisper, misma calidad |
| Traducción | NLLB-200 distilled-1.3B (Meta) | 200 idiomas, calidad alta, 100% offline |
| UI | Textual | TUI rica multiplataforma sin Qt/Tk |
| GPU | CUDA / MPS / CPU autodetectado | Soporta Blackwell (cu128) |

## Requisitos

- Python **3.12+**
- **Windows**: nada extra; WASAPI loopback es nativo
- **Linux**: `pactl` (paquete `pulseaudio-utils`)
- **macOS**: `brew install blackhole-2ch` + Multi-Output Device en Audio MIDI Setup
- **GPU NVIDIA**: drivers recientes
  - Blackwell (RTX 50-series, sm_120) requiere **PyTorch cu128** (incluido en el instalador)
  - Pre-Blackwell funciona también con cu128

## Instalación

### Windows (PowerShell)

```powershell
cd C:\Users\crist\Documents\realtime-transcriber
.\scripts\install_windows.ps1
```

### Linux

```bash
cd realtime-transcriber
./scripts/install_linux.sh
```

### macOS

```bash
cd realtime-transcriber
./scripts/install_macos.sh
```

Los scripts crean `.venv`, instalan PyTorch con CUDA 12.8 (Blackwell-compatible)
y todas las dependencias, y dejan el paquete en modo editable.

## Pre-descarga de modelos (recomendado)

Para que el primer arranque no sea sorprendente (~8 GB total), descarga todos
los modelos por adelantado:

```powershell
. .\.venv\Scripts\Activate.ps1
python scripts\download_models.py
```

## Comando "doctor"

Diagnóstico de entorno (Python, GPU, librerías, dispositivos audio, caché):

```powershell
python scripts\doctor.py
```

## Uso

### Modo GUI (ventana flotante translúcida) — recomendado

```powershell
transcriber-gui            # ventana Qt frameless con efecto cristal
# o equivalente:
transcriber --gui
```

Características de la GUI:
- Ventana **frameless** con efecto cristal oscuro (dark glass).
- **Redimensionable** desde cualquier borde o esquina.
- **Arrastrable** desde la barra de título.
- **Slider de opacidad** en vivo (60-100%).
- **Pin** para mantener siempre encima.
- Dos paneles separados: **Original** (texto fuente) y **Español** (traducción).
- Cada panel muestra los segmentos finales + un **caption en vivo** en itálica/gris que se actualiza cada 1.5s mientras el speaker habla.
- Strip inferior con métricas en tiempo real: segmentos, RTF, latencia E2E, **uso de CPU**, **uso de GPU**, y **VRAM** usada/total.

### Modo TUI (terminal)

```powershell
transcriber                # TUI Textual en terminal
transcriber --debug        # logs DEBUG a logs/transcriber.log
transcriber --no-translate # transcribe pero no traduce
```

Atajos TUI:
- `q` — salir
- `c` — limpiar transcripción

### Test de captura aislada

```powershell
transcriber-capture-test --seconds 30 --output capture_test.wav
```

### Tests offline reproducibles

```powershell
python scripts\test_vad_segmenter.py capture_test.wav
python scripts\test_offline_asr.py    capture_test.wav
python scripts\test_translation.py
```

## Configuración (`.env`)

Copia `.env.example` a `.env`. Variables relevantes:

| Variable | Default | Descripción |
|---|---|---|
| `TRANSCRIBER_COMPUTE_DEVICE` | `auto` | `auto` / `cuda` / `mps` / `cpu` |
| `TRANSCRIBER_WHISPER_MODEL` | `large-v3` | tiny / base / small / medium / large-v2 / large-v3 |
| `TRANSCRIBER_TRANSLATION_BACKEND` | `nllb-1.3b` | `nllb-1.3b` / `nllb-600m` |
| `TRANSCRIBER_AUDIO_DEVICE_HINT` | (vacío) | substring del nombre del dispositivo a forzar |
| `TRANSCRIBER_TARGET_LANGUAGE` | `es` | ISO 639-1 del idioma destino |
| `TRANSCRIBER_VAD_AGGRESSIVENESS` | `0.5` | 0 (laxo) – 1 (estricto) |
| `TRANSCRIBER_MIN_SEGMENT_MS` | `1500` | mínimo de un segmento |
| `TRANSCRIBER_MAX_SEGMENT_MS` | `14000` | máximo antes de force-cut |
| `TRANSCRIBER_SILENCE_TAIL_MS` | `400` | silencio para cerrar un segmento |
| `TRANSCRIBER_PREVIEW_ENABLED` | `true` | activar live captions |
| `TRANSCRIBER_PREVIEW_INTERVAL_MS` | `2500` | cada cuánto refrescar el preview |
| `TRANSCRIBER_LOG_LEVEL` | `INFO` | nivel de log |

## Arquitectura

```
src/transcriber/
├── audio/          # WASAPI / Pulse / CoreAudio + autoselector + recovery
├── pipeline/       # ring buffer, VAD, segmenter, orchestrator
├── asr/            # faster-whisper wrapper (con modo preview)
├── translation/    # NLLB-200 (con tabla Whisper→NLLB de 80+ idiomas)
├── ui/             # TUI Textual con live captions
├── hardware/       # detección CUDA / MPS / CPU
├── config.py       # pydantic-settings, .env
└── logging_setup.py
```

### Threading model

1. **PortAudio callback** → cola de chunks
2. **capture-pump** → VU meter + forward al segmenter
3. **segmenter** → VAD + segmentación → cola ASR
4. **asr** → Whisper final + traducción NLLB → eventos UI
5. **asr-preview** → cada 2.5s peek al segmento abierto, transcribe rápido + traduce → eventos UI
6. **UI** (Textual) → consume eventos asyncio

### Auto-recovery

Si el stream de captura falla (cambio de dispositivo por defecto, pausa del
servicio de audio, etc.) el supervisor reintenta con backoff exponencial
(1, 2, 4, 8, 16 s) antes de dar error fatal.

## Tests

```powershell
pytest -q
```

## Solución de problemas

**CUDA no disponible (`cuda available: False`)**
Reinstala torch desde el índice CUDA correcto:
```powershell
pip uninstall -y torch torchaudio
pip install --index-url https://download.pytorch.org/whl/cu128 torch torchaudio
```

**WASAPI sin dispositivos loopback**
- Asegúrate que el servicio "Windows Audio" está corriendo.
- Reinstala PyAudioWPatch: `pip install --force-reinstall PyAudioWPatch`.

**Linux: `pactl` no encontrado**
```bash
sudo apt install pulseaudio-utils
```

**macOS: "No loopback device found"**
```bash
brew install blackhole-2ch
```
Luego en Audio MIDI Setup crea un Multi-Output Device combinando BlackHole 2ch
y tus altavoces, y selecciónalo como salida del sistema.

**Latencia alta**
- Si supera 15s consistentemente, sube `TRANSCRIBER_PREVIEW_INTERVAL_MS` a `2000` o baja `TRANSCRIBER_MAX_SEGMENT_MS` a `10000`.
- Verifica con `nvidia-smi` que la GPU se esté usando (memoria > 4 GB durante operación).

**RTF > 1.0**
Whisper no está en GPU. Ejecuta `python scripts\doctor.py` para confirmar.

## Estado

| Hito | Descripción | Estado |
|------|-------------|--------|
| 1 | Captura cross-platform | ✅ |
| 2 | VAD + segmentación | ✅ |
| 3 | ASR offline (faster-whisper large-v3) | ✅ |
| 4 | ASR en vivo + TUI Textual | ✅ |
| 5 | Traducción NLLB-200 1.3B | ✅ |
| 5+ | Live preview captions | ✅ |
| 6 | Hardening + packaging | ✅ |

## Licencia

MIT.
