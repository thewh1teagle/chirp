import { convertFileSrc, invoke } from "@tauri-apps/api/core";
import { open } from "@tauri-apps/plugin-dialog";
import { useEffect, useMemo, useState } from "react";
import "./App.css";

type ModelBundle = {
  installed: boolean;
  model_path: string;
  codec_path: string;
  model_dir: string;
  version: string;
  url: string;
};

type RunnerInfo = {
  base_url: string;
};

const sampleText =
  "Chirp is running locally through a bundled sidecar. This voice was generated from the Qwen three text to speech model.";

function App() {
  const [bundle, setBundle] = useState<ModelBundle | null>(null);
  const [runnerUrl, setRunnerUrl] = useState("");
  const [text, setText] = useState(sampleText);
  const [referencePath, setReferencePath] = useState("");
  const [languages, setLanguages] = useState<string[]>(["auto"]);
  const [language, setLanguage] = useState("auto");
  const [audioPath, setAudioPath] = useState("");
  const [status, setStatus] = useState("Checking model bundle...");
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState("");

  const audioSrc = useMemo(() => (audioPath ? convertFileSrc(audioPath) : ""), [audioPath]);

  useEffect(() => {
    refreshModel();
  }, []);

  async function refreshModel() {
    try {
      const current = await invoke<ModelBundle>("get_model_bundle");
      setBundle(current);
      setStatus(current.installed ? "Model bundle is installed." : "Model bundle is not installed.");
    } catch (err) {
      setError(String(err));
      setStatus("Could not inspect the model bundle.");
    }
  }

  async function downloadModel() {
    setBusy(true);
    setError("");
    setStatus("Downloading model bundle. This is about 1.18 GB and may take a while.");
    try {
      const installed = await invoke<ModelBundle>("download_model_bundle");
      setBundle(installed);
      setStatus("Model bundle is installed.");
    } catch (err) {
      setError(String(err));
      setStatus("Model download failed.");
    } finally {
      setBusy(false);
    }
  }

  async function startRunner() {
    setBusy(true);
    setError("");
    setStatus("Starting runner...");
    try {
      const info = await invoke<RunnerInfo>("start_runner");
      setRunnerUrl(info.base_url);
      setStatus("Runner is ready.");
      return info.base_url;
    } catch (err) {
      setError(String(err));
      setStatus("Runner failed to start.");
      throw err;
    } finally {
      setBusy(false);
    }
  }

  async function chooseReference() {
    const selected = await open({
      multiple: false,
      filters: [
        {
          name: "WAV audio",
          extensions: ["wav"],
        },
      ],
    });
    if (typeof selected === "string") {
      setReferencePath(selected);
    }
  }

  async function synthesize() {
    const current = bundle ?? (await invoke<ModelBundle>("get_model_bundle"));
    setBundle(current);
    if (!current.installed) {
      setStatus("Install the model bundle before synthesizing.");
      return;
    }
    if (!text.trim()) {
      setStatus("Enter text to synthesize.");
      return;
    }

    setBusy(true);
    setError("");
    setAudioPath("");
    setStatus("Loading model...");
    try {
      const info = await invoke<RunnerInfo>("start_runner");
      setRunnerUrl(info.base_url);
      await invoke("load_model", {
        request: {
          model_path: current.model_path,
          codec_path: current.codec_path,
          temperature: 0.9,
          top_k: 50,
        },
      });
      const supportedLanguages = await invoke<string[]>("get_languages");
      setLanguages(supportedLanguages.length ? supportedLanguages : ["auto"]);
      const selectedLanguage = supportedLanguages.includes(language) ? language : "auto";
      if (selectedLanguage !== language) {
        setLanguage("auto");
      }
      setStatus("Synthesizing WAV...");
      const output = await invoke<string>("synthesize", {
        request: {
          input: text,
          voice_reference: referencePath || null,
          language: selectedLanguage,
        },
      });
      setAudioPath(output);
      setStatus("Synthesis complete.");
    } catch (err) {
      setError(String(err));
      setStatus("Synthesis failed.");
    } finally {
      setBusy(false);
    }
  }

  return (
    <main className="app-shell">
      <section className="workspace">
        <header className="topbar">
          <div>
            <h1>Chirp</h1>
            <p>Local Qwen3-TTS synthesis</p>
          </div>
          <button type="button" onClick={startRunner} disabled={busy}>
            Start Runner
          </button>
        </header>

        <section className="status-grid">
          <div>
            <span>Model</span>
            <strong>{bundle?.installed ? bundle.version : "Not installed"}</strong>
          </div>
          <div>
            <span>Runner</span>
            <strong>{runnerUrl || "Stopped"}</strong>
          </div>
        </section>

        <section className="model-panel">
          <div>
            <h2>Model Bundle</h2>
            <p>{bundle?.model_dir || "Checking app data directory..."}</p>
          </div>
          <button type="button" onClick={downloadModel} disabled={busy || bundle?.installed}>
            {bundle?.installed ? "Installed" : "Download Q5_0"}
          </button>
        </section>

        <section className="synth-panel">
          <label htmlFor="text">Text</label>
          <textarea id="text" value={text} onChange={(event) => setText(event.currentTarget.value)} />
          <div className="reference-row">
            <div>
              <span>Reference WAV</span>
              <strong>{referencePath || "Default voice"}</strong>
            </div>
            <div className="reference-actions">
              <button type="button" onClick={chooseReference} disabled={busy}>
                Choose WAV
              </button>
              <button type="button" className="secondary" onClick={() => setReferencePath("")} disabled={busy || !referencePath}>
                Clear
              </button>
            </div>
          </div>
          <div className="language-row">
            <label htmlFor="language">Language</label>
            <select id="language" value={language} onChange={(event) => setLanguage(event.currentTarget.value)} disabled={busy}>
              {languages.map((item) => (
                <option key={item} value={item}>
                  {item}
                </option>
              ))}
            </select>
          </div>
          <div className="actions">
            <button type="button" onClick={synthesize} disabled={busy || !text.trim()}>
              Synthesize
            </button>
            <span>{status}</span>
          </div>
        </section>

        {audioSrc && (
          <section className="audio-panel">
            <audio src={audioSrc} controls />
            <p>{audioPath}</p>
          </section>
        )}

        {error && <pre className="error">{error}</pre>}
      </section>
    </main>
  );
}

export default App;
