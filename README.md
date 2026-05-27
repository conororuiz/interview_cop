# Realtime System-Audio Transcriber + Translator + Interview Copilot

Captura el audio que reproduce tu sistema (podcasts, vídeos, llamadas, streams,
entrevistas) y lo transcribe en tiempo real. Si el idioma detectado no es
español, lo traduce automáticamente. Opcionalmente, un copiloto de entrevista
basado en **Gemini** te sugiere — en primera persona y en el idioma del audio —
qué responder a las preguntas que se hagan.

Pensado para precisión por delante de consumo de recursos, con baja latencia
gracias a transcripción de vista previa y traducción en paralelo.

## Stack tecnológico

| Capa | Tecnología | Por qué |
|---|---|---|
| Captura Windows | PyAudioWPatch (WASAPI loopback) | Loopback nativo sin cables virtuales |
| Captura Linux | sounddevice + PulseAudio/PipeWire monitor | Estándar en escritorios Linux modernos |
| Captura macOS | sounddevice + BlackHole | CoreAudio no expone loopback nativo |
| VAD | Silero VAD v5 (paquete oficial) | Mejor precisión que WebRTC en habla con música/ruido |
| ASR | faster-whisper large-v3 (CTranslate2) | 4-5× más rápido que openai-whisper, misma calidad |
| Traducción | NLLB-200 distilled-1.3B (Meta) | 200 idiomas, calidad alta, 100% offline |
| Copiloto IA | Google Gemini (streaming) | Respuestas en tiempo real en modo entrevista |
| GUI | CustomTkinter | Ligero, sin Qt, sin issues de compositor en Blackwell |
| TUI | Textual + Rich | Modo terminal multiplataforma |
| GPU | CUDA / MPS / CPU autodetectado | Soporta Blackwell (cu128) y fallback a CPU |

## Requisitos

- Python **3.12+**
- **Windows**: nada extra; WASAPI loopback es nativo.
- **Linux**: `pactl` (paquete `pulseaudio-utils`).
- **macOS**: `brew install blackhole-2ch` + Multi-Output Device en Audio MIDI Setup.
- **GPU NVIDIA** (recomendado para uso en vivo):
  - Blackwell (RTX 50-series, sm_120) requiere **PyTorch cu128** (incluido en el instalador).
  - Pre-Blackwell también funciona con cu128.
- **Sin GPU**: usa `--cpu` (ver más abajo). Funciona, solo más lento.
- **Para el copiloto IA**: una API key gratuita de Google Gemini.

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

Los scripts crean `.venv`, instalan PyTorch con la **rueda correcta para tu
máquina** (cu128 si detectan una GPU NVIDIA con `nvidia-smi`, CPU si no la hay),
todas las dependencias, dejan el paquete en modo editable y al final ejecutan
`scripts/autoconfig.py` para escribir un `.env` afinado al hardware del equipo.

## Autoconfiguración del hardware

Cada equipo es distinto. Para evitar que el primer arranque reviente por una
configuración pensada para otra máquina, el instalador ejecuta al final
**`scripts/autoconfig.py`**, que detecta:

- Si hay GPU NVIDIA (vía `nvidia-smi`) y cuánta VRAM tiene.
- Si hay GPU AMD/Radeon (vía `lspci` en Linux, `Get-CimInstance` en Windows,
  o `rocm-smi` si está instalado) — y la nombra explícitamente aunque caiga
  a CPU.
- Si es Apple Silicon (M1/M2/M3 → MPS).
- La cantidad de RAM y de cores del CPU.

…y escribe un `.env` con el modelo de Whisper, el backend de traducción,
`compute_type` (fp16 / int8_fp16 / int8) y los tiempos de segmentación
adecuados para ese tier:

| Tier | Hardware típico | Whisper | NLLB | Compute | `MAX_SEGMENT_MS` |
|---|---|---|---|---|---|
| GPU-HIGH | ≥12 GB VRAM (RTX 5070/4090…) | large-v3 | 1.3B | float16 | 10000 |
| GPU-UPPER-MID | 8-11 GB VRAM (RTX 3070/4060 Ti) | large-v3 | 1.3B | int8_float16 | 10000 |
| GPU-MID | 6-7 GB VRAM (RTX 3060/4050) | medium | 1.3B | float16 | 10000 |
| GPU-LOW | 4-5 GB VRAM | small | 600M | int8_float16 | 12000 |
| GPU-VERY-LOW | 2-3 GB VRAM (GTX 1060 3GB) | small | 600M | int8 | 14000 |
| APPLE-HIGH | M-series ≥16 GB | medium (MPS) | 1.3B | — | 12000 |
| APPLE-LOW | M-series 8 GB | small (MPS) | 600M | — | 12000 |
| CPU-HIGH | AMD/Radeon o sin GPU, ≥16 GB RAM | medium | 600M | int8 | 12000 |
| CPU-MID | AMD/Radeon o sin GPU, 8-15 GB RAM | small | 600M | int8 | 12000 |
| CPU-LOW | AMD/Radeon o sin GPU, <8 GB RAM | base | 600M | int8 | 14000 |

### ¿Por qué AMD/Radeon cae a CPU?

faster-whisper (motor CTranslate2) sólo tiene backend CUDA — no existe
build para ROCm. Aunque instaláramos PyTorch con ROCm en Linux, sólo se
aceleraría NLLB (la traducción); Whisper seguiría en CPU. Además, ROCm
oficialmente sólo soporta RDNA2 o superior (RX 6000+), así que una
RX 5600 / 5700 (RDNA1), Polaris (RX 400/500) o Vega ni eso. Por eso el
instalador detecta tu Radeon, te la muestra por su nombre en el reporte
y elige el mejor tier CPU para tu RAM — no rompe la instalación.

Si estás en Linux con una RX 6000/7000 y quieres acelerar NLLB
manualmente:

```bash
pip install --index-url https://download.pytorch.org/whl/rocm6.2 torch torchaudio
# luego en .env:
#   TRANSCRIBER_COMPUTE_DEVICE=cuda    (ROCm se hace pasar por "cuda" en torch)
# pero deja TRANSCRIBER_COMPUTE_TYPE=int8 para que Whisper siga funcionando en CPU
```

Puedes re-ejecutarlo manualmente cuando quieras (por ejemplo, después de
cambiar de tarjeta gráfica):

```powershell
python scripts\autoconfig.py            # detecta y escribe .env
python scripts\autoconfig.py --dry-run  # solo muestra el plan, no toca nada
python scripts\autoconfig.py --force    # sobrescribe valores que tú hubieras puesto a mano
python scripts\autoconfig.py --json     # salida machine-readable
```

Los valores existentes en `.env` se **respetan por defecto**: si tú fijaste
`TRANSCRIBER_WHISPER_MODEL=large-v3`, el autoconfig no lo va a tocar a menos
que uses `--force`. Las API keys (`GEMINI_API_KEY`, `DEEPL_API_KEY`) nunca
las modifica.

## Pre-descarga de modelos (recomendado)

Para que el primer arranque no sea sorprendente (~8 GB total con GPU,
o ~3 GB con `--cpu`), descarga los modelos por adelantado:

```powershell
. .\.venv\Scripts\Activate.ps1
python scripts\download_models.py
```

## Configuración del archivo `.env`

Copia `.env.example` a `.env` en la raíz del proyecto. El archivo debe llamarse
**exactamente** `.env` (sin `.txt` ni nada al final — cuidado con Windows que
oculta extensiones por defecto).

### Setup mínimo del copiloto IA (Gemini)

Para que el botón **"✨ Responder"** funcione necesitas una API key gratuita:

1. Ve a https://aistudio.google.com/app/apikey (cuenta Google, sin tarjeta).
2. Crea una API key.
3. Pégala en `.env`:

   ```
   GEMINI_API_KEY=AIzaSy...tu-key-completa
   ```

> **Importante**: la variable se llama **`GEMINI_API_KEY`** (sin prefijo
> `TRANSCRIBER_`), porque es el nombre convencional que usan Google y muchas
> otras herramientas. Si la nombras de otra forma, el botón seguirá
> deshabilitado.

> **Formato**: no pongas espacios alrededor del `=`, no uses comillas, no
> añadas `export ` delante. Solo `GEMINI_API_KEY=...` en una línea.

Para verificar que la key se está leyendo, busca esta línea en los logs al
arrancar:

```
INFO | transcriber.ai.gemini_engine | Gemini responder ready (model=gemini-2.5-flash, key=...XYZ4).
```

Si en su lugar ves `GEMINI_API_KEY not found in environment`, revisa:

- Que el archivo se llame `.env` (no `.env.txt`).
- Que esté en la raíz del proyecto (mismo directorio que `pyproject.toml`).
- Que la línea sea `GEMINI_API_KEY=AIza...` sin espacios extras.

Verifica desde una consola:

```powershell
python -c "import os; from dotenv import load_dotenv; load_dotenv('.env'); print('key:', (os.getenv('GEMINI_API_KEY') or 'NOT FOUND')[:10] + '...')"
```

### Variables principales del `.env`

| Variable | Default | Descripción |
|---|---|---|
| `GEMINI_API_KEY` | (sin default) | **Sin prefijo**. API key de Google Gemini para el copiloto IA |
| `TRANSCRIBER_GEMINI_MODEL` | `gemini-2.5-flash` | Modelo Gemini (también `gemini-2.5-pro`, etc.) |
| `TRANSCRIBER_AI_HISTORY_SECONDS` | `60` | Ventana de transcripción enviada como contexto a Gemini |
| `TRANSCRIBER_COMPUTE_DEVICE` | `auto` | `auto` / `cuda` / `mps` / `cpu` |
| `TRANSCRIBER_WHISPER_MODEL` | `large-v3` | tiny / base / small / medium / large-v2 / large-v3 |
| `TRANSCRIBER_TRANSLATION_BACKEND` | `nllb-1.3b` | `nllb-1.3b` / `nllb-600m` |
| `TRANSCRIBER_AUDIO_DEVICE_HINT` | (vacío) | Substring del nombre del dispositivo a forzar |
| `TRANSCRIBER_TARGET_LANGUAGE` | `es` | ISO 639-1 del idioma destino para la traducción |
| `TRANSCRIBER_VAD_AGGRESSIVENESS` | `0.5` | 0 (laxo) – 1 (estricto) |
| `TRANSCRIBER_MIN_SEGMENT_MS` | `1500` | Mínimo de un segmento |
| `TRANSCRIBER_MAX_SEGMENT_MS` | `10000` | Máximo antes de force-cut |
| `TRANSCRIBER_SILENCE_TAIL_MS` | `300` | Silencio para cerrar un segmento |
| `TRANSCRIBER_PREVIEW_ENABLED` | `true` | Activar live captions |
| `TRANSCRIBER_PREVIEW_INTERVAL_MS` | `1500` | Cada cuánto refrescar el preview |
| `TRANSCRIBER_PREVIEW_MIN_AUDIO_MS` | `1500` | Audio mínimo acumulado antes del primer preview |
| `TRANSCRIBER_LOG_LEVEL` | `INFO` | Nivel de log |

## Comando "doctor"

Diagnóstico de entorno (Python, GPU, librerías, dispositivos audio, caché):

```powershell
python scripts\doctor.py
```

## Uso

### Modo GUI (ventana flotante) — recomendado

```powershell
transcriber-gui            # CustomTkinter (ligero, default)
transcriber --gui          # equivalente
transcriber-gui --qt       # versión PySide6 (experimental, más pesada)
```

Características de la GUI:

- Ventana **flotante** con frame nativo de Windows, semi-transparente (94%).
- **Redimensionable** desde cualquier borde.
- **Arrastrable** desde la barra de título.
- **Slider de opacidad** en vivo (60-100%).
- **Pin** para mantener siempre encima (`📌`).
- **Tres paneles** verticales:
  - **Original**: transcripción del audio fuente con captions en vivo.
  - **Español**: traducción automática (oculto si el audio ya está en español).
  - **Tu respuesta**: sugerencia del copiloto IA cuando pulses **"✨ Responder"**.
- Strip de métricas en tiempo real: segmentos, RTF medio/último, latencia E2E,
  CPU%, GPU%, VRAM usada/total.
- Botón **🗑 Limpiar**: borra los tres paneles **y** el historial de contexto
  de la IA (importante: el siguiente "Responder" solo verá lo que llegue
  desde ese momento).

### Modo entrevista (copiloto IA)

Con la API key configurada en `.env` y la app arrancada, durante una entrevista:

1. Reproduces el audio (Zoom, Meet, Teams, etc.) que el entrevistador genera.
2. La transcripción aparece en el panel **Original** (y traducida en **Español**).
3. Cuando el entrevistador hace una pregunta, pulsas **"✨ Responder"**.
4. El panel **Tu respuesta** muestra, en streaming y **en el idioma del audio**,
   lo que dirías como entrevistado. Habla en primera persona, usa STAR para
   preguntas behavioural ("describe a time when…") y deja placeholders como
   `[mi empresa actual]` o `[un proyecto reciente]` para que tú los rellenes.

### Modo `--cpu` (simular sin GPU)

Útil para ver cómo se comporta la app en una máquina sin GPU:

```powershell
transcriber-gui --cpu
```

Esto:

- Fuerza `compute_device=cpu` aunque tengas CUDA disponible.
- Auto-baja los modelos a tamaños usables en CPU si no los has fijado en `.env`:
  - **Whisper medium** (~1.5 GB, RTF ~0.7-1.5 en CPU)
  - **NLLB-200 600M** (~2.4 GB, traducciones en ~3-5s)
- Ensancha los segmentos a 12s y los previews a cada 3s para amortizar el
  coste por llamada en CPU.

Comparativa orientativa:

| Métrica | RTX 5070 (GPU) | CPU típico (8-core) |
|---|---|---|
| Tiempo a "Ready" | ~10-15s | ~15-30s |
| RTF Whisper | ~0.15 | ~0.7-1.5 (medium) |
| Latencia traducción | ~1.5s | ~3-5s |
| Latencia E2E típica | 2-4s | 8-15s |
| VRAM usada | ~5 GB | 0 |
| RAM usada | ~1 GB | ~4-6 GB |

Si quieres comparar la misma calidad que con GPU (más lento aún):

```powershell
transcriber-gui --cpu --whisper-model large-v3 --translator nllb-1.3b
```

### Otros flags útiles

```powershell
transcriber-gui --no-translate           # solo transcribir, sin NLLB
transcriber-gui --debug                  # logs DEBUG a logs/transcriber.log
transcriber-gui --whisper-model small    # forzar modelo Whisper
transcriber-gui --translator nllb-600m   # forzar traductor
transcriber-gui --cpu --no-translate --whisper-model small   # modo más liviano posible
```

### Modo TUI (terminal)

```powershell
transcriber                # TUI Textual en terminal
transcriber --debug
transcriber --no-translate
transcriber --cpu
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

## Arquitectura

```
src/transcriber/
├── audio/          # WASAPI / Pulse / CoreAudio + autoselector + recovery
├── pipeline/       # ring buffer, VAD, segmenter, orchestrator
├── asr/            # faster-whisper wrapper (con modo preview)
├── translation/    # NLLB-200 (con tabla Whisper→NLLB de 80+ idiomas)
├── ai/             # Copiloto Gemini (modo entrevista, streaming)
├── ui/             # gui_ctk (CustomTkinter, default), gui (PySide6), tui (Textual)
├── hardware/       # detección CUDA / MPS / CPU + monitor CPU/GPU/VRAM en vivo
├── config.py       # pydantic-settings + carga eager de .env
└── logging_setup.py
```

### Threading model

1. **PortAudio callback** → cola de chunks
2. **capture-pump** → VU meter + forward al segmenter
3. **segmenter** → VAD + segmentación → cola ASR
4. **asr** → Whisper final → para no-español: enqueue a cola de traducción
5. **translation** (hilo dedicado) → NLLB → emite `EvtTranscript` con traducción
6. **asr-preview** → cada 1.5s peek al segmento abierto, transcribe rápido + traduce → eventos UI
7. **ai-responder** → consume requests del usuario, llama a Gemini en streaming
8. **UI** (Tkinter mainloop o Textual o Qt) → consume eventos via `root.after`

### Auto-recovery

Si el stream de captura falla (cambio de dispositivo por defecto, pausa del
servicio de audio, etc.) el supervisor reintenta con backoff exponencial
(1, 2, 4, 8, 16 s) antes de dar error fatal.

## Tests

```powershell
pytest -q
```

## Solución de problemas

**El botón "✨ Responder" sale deshabilitado**

- Asegúrate de tener `GEMINI_API_KEY=...` en `.env` (sin prefijo, sin comillas).
- Verifica que la key se lee:
  ```powershell
  python -c "import os; from dotenv import load_dotenv; load_dotenv('.env'); print((os.getenv('GEMINI_API_KEY') or 'NOT FOUND')[:10])"
  ```
- Reinicia la app después de modificar `.env`.

**CUDA no disponible (`cuda available: False`)**

Reinstala torch desde el índice CUDA correcto:

```powershell
pip uninstall -y torch torchaudio
pip install --index-url https://download.pytorch.org/whl/cu128 torch torchaudio
```

**WASAPI sin dispositivos loopback (Windows)**

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

- Si supera 15s consistentemente, sube `TRANSCRIBER_PREVIEW_INTERVAL_MS` a `2000`
  o baja `TRANSCRIBER_MAX_SEGMENT_MS` a `8000`.
- Verifica con `nvidia-smi` que la GPU se esté usando (memoria > 4 GB durante operación).
- Si estás en `--cpu`, espera latencias de 8-15s — es lo esperado.

**RTF > 1.0**

Whisper no está en GPU. Ejecuta `python scripts\doctor.py` para confirmar.

**La ventana de la GUI sale negra o congelada (Qt)**

Cambia a la GUI ligera de CustomTkinter (default desde la versión actual):

```powershell
transcriber-gui          # CustomTkinter (no Qt)
```

## Estado

| Hito | Descripción | Estado |
|------|-------------|--------|
| 1 | Captura cross-platform | ✅ |
| 2 | VAD + segmentación | ✅ |
| 3 | ASR offline (faster-whisper large-v3) | ✅ |
| 4 | ASR en vivo + TUI Textual | ✅ |
| 5 | Traducción NLLB-200 1.3B | ✅ |
| 5+ | Live preview captions | ✅ |
| 5++ | Pipeline paralelo de traducción | ✅ |
| 6 | Hardening + packaging | ✅ |
| 7 | GUI CustomTkinter (ligera) | ✅ |
| 8 | Copiloto IA Gemini en modo entrevista | ✅ |
| 9 | Modo `--cpu` para simular ejecución sin GPU | ✅ |
| 10 | Autoconfig de hardware al instalar (tiers VRAM/RAM) | ✅ |

## Licencia

MIT.
