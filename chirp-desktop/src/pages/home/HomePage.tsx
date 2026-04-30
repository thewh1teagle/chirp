import { convertFileSrc, invoke } from "@tauri-apps/api/core";
import { open } from "@tauri-apps/plugin-dialog";
import { AnimatePresence, motion } from "framer-motion";
import { useEffect, useMemo, useRef } from "react";
import type { Dispatch, SetStateAction } from "react";
import { useNavigate } from "react-router-dom";
import type { ModelBundle, RunnerInfo, StudioState } from "../../lib/types";
import { AppFrame } from "../../components/AppFrame";
import { CreateStatus } from "../../components/CreateStatus";
import { ErrorBlock } from "../../components/ui";
import { WaveformPlayer } from "../../components/WaveformPlayer";
import { StudioHeader } from "../../components/WorkspaceHeader";
import { EditorCard } from "./EditorCard";
import { VoiceSettings } from "./VoiceSettings";

type PageProps = {
  bundle: ModelBundle | null;
  setBundle: (bundle: ModelBundle) => void;
};

type HomePageProps = PageProps & {
  studio: StudioState;
  setStudio: Dispatch<SetStateAction<StudioState>>;
};

export function HomePage({ bundle, setBundle, studio, setStudio }: HomePageProps) {
  const navigate = useNavigate();
  const { text, referencePath, languages, language, kokoroVoice, kokoroVoiceIds, audioPath, audioAutoplayPending, step, status, busy, error } = studio;
  const loadingLanguagesRef = useRef(false);

  const audioSrc = useMemo(() => (audioPath ? convertFileSrc(audioPath) : ""), [audioPath]);
  const updateStudio = (patch: Partial<StudioState>) => setStudio((current) => ({ ...current, ...patch }));

  useEffect(() => {
    const needsKokoroVoices = bundle?.runtime === "kokoro" && kokoroVoiceIds.length === 0;
    if (!bundle?.installed || busy || loadingLanguagesRef.current) return;
    if (languages.length > 1 && !needsKokoroVoices) return;
    const currentBundle = bundle;
    loadingLanguagesRef.current = true;

    async function loadLanguages() {
      try {
        await invoke<RunnerInfo>("start_runner");
        await invoke("load_model", {
          request: {
            runtime: currentBundle.runtime,
            model_path: currentBundle.model_path,
            codec_path: currentBundle.codec_path || "",
            voices_path: currentBundle.voices_path || "",
            espeak_data_path: currentBundle.espeak_data_path || "",
            voice: currentBundle.runtime === "kokoro" ? kokoroVoice : "af_heart",
            temperature: 0.9,
            top_k: 50,
          },
        });
        const supportedLanguages = await invoke<string[]>("get_languages");
        const nextStudio: Partial<StudioState> = {};
        if (supportedLanguages.length) nextStudio.languages = supportedLanguages;
        if (currentBundle.runtime === "kokoro") {
          try {
            const voiceIds = await invoke<string[]>("get_voices");
            if (voiceIds.length) nextStudio.kokoroVoiceIds = voiceIds;
          } catch {
            // Voice IDs improve the picker, but language loading should still succeed without them.
          }
        }
        if (Object.keys(nextStudio).length) updateStudio(nextStudio);
      } catch {
        updateStudio({ languages: ["auto"] });
      } finally {
        loadingLanguagesRef.current = false;
      }
    }

    loadLanguages();
  }, [bundle, busy, languages.length, kokoroVoiceIds.length]);

  async function chooseReference() {
    const selected = await open({
      multiple: false,
      filters: [{ name: "WAV audio", extensions: ["wav"] }],
    });
    if (typeof selected === "string") updateStudio({ referencePath: selected });
  }

  async function createVoice() {
    const preferredRuntime = bundle?.runtime ?? localStorage.getItem("chirp.runtime") ?? "qwen";
    const current = bundle ?? (await invoke<ModelBundle>("get_model_bundle_for_runtime", { runtime: preferredRuntime }));
    setBundle(current);
    if (!current.installed) {
      navigate("/onboard", { replace: true });
      return;
    }
    if (!text.trim()) {
      updateStudio({ status: "Input text required." });
      return;
    }

    updateStudio({ busy: true, error: "", audioPath: "", audioAutoplayPending: false });
    try {
      updateStudio({ step: "starting", status: "Initializing Engine..." });
      await invoke<RunnerInfo>("start_runner");

      updateStudio({ step: "loading", status: "Loading models..." });
      await invoke("load_model", {
        request: {
          runtime: current.runtime,
          model_path: current.model_path,
          codec_path: current.codec_path || "",
          voices_path: current.voices_path || "",
          espeak_data_path: current.espeak_data_path || "",
          voice: current.runtime === "kokoro" ? kokoroVoice : "af_heart",
          temperature: 0.9,
          top_k: 50,
        },
      });

      const supportedLanguages = await invoke<string[]>("get_languages");
      const nextStudio: Partial<StudioState> = { languages: supportedLanguages.length ? supportedLanguages : ["auto"] };
      if (current.runtime === "kokoro") {
        try {
          const voiceIds = await invoke<string[]>("get_voices");
          if (voiceIds.length) nextStudio.kokoroVoiceIds = voiceIds;
        } catch {
          // Keep synthesis usable even if voice listing is unavailable.
        }
      }
      updateStudio(nextStudio);
      const selectedLanguage = current.runtime === "kokoro" ? "auto" : supportedLanguages.includes(language) ? language : "auto";
      if (selectedLanguage !== language) updateStudio({ language: "auto" });

      updateStudio({ step: "creating", status: "Generating audio..." });
      const output = await invoke<string>("synthesize", {
        request: {
          input: text,
          voice_reference: current.runtime === "kokoro" ? null : referencePath || null,
          voice: current.runtime === "kokoro" ? kokoroVoice : null,
          language: selectedLanguage,
        },
      });
      updateStudio({ audioPath: output, audioAutoplayPending: true, step: "done", status: "Generation complete." });
    } catch (err) {
      updateStudio({ step: "idle", error: String(err), status: "Generation failed." });
    } finally {
      updateStudio({ busy: false });
    }
  }

  return (
    <AppFrame bundle={bundle}>
      <div className="w-full max-w-[1200px]">
        <StudioHeader bundle={bundle} />
        <div className="grid gap-12 mt-4 lg:grid-cols-[1fr_360px]">
          <div className="space-y-6">
            <EditorCard
              busy={busy}
              text={text}
              setText={(nextText) => updateStudio({ text: nextText })}
              createVoice={createVoice}
            />

            <AnimatePresence>
              {audioPath && (
                <motion.div initial={{ opacity: 0, y: 12 }} animate={{ opacity: 1, y: 0 }} exit={{ opacity: 0, y: 8 }}>
                  <WaveformPlayer
                    src={audioSrc}
                    sourcePath={audioPath}
                    filename={audioPath.split(/[\\/]/).pop() || "generated-audio.wav"}
                    autoPlayOnce={audioAutoplayPending}
                    onAutoPlayConsumed={() => updateStudio({ audioAutoplayPending: false })}
                  />
                </motion.div>
              )}
            </AnimatePresence>
          </div>

          <aside className="space-y-6">
            <VoiceSettings
              busy={busy}
              language={language}
              languages={languages}
              referencePath={referencePath}
              runtime={bundle?.runtime ?? "qwen"}
              kokoroVoice={kokoroVoice}
              kokoroVoiceIds={kokoroVoiceIds}
              chooseReference={chooseReference}
              setLanguage={(nextLanguage) => updateStudio({ language: nextLanguage })}
              setKokoroVoice={(nextVoice) => updateStudio({ kokoroVoice: nextVoice })}
              setReferencePath={(nextPath) => updateStudio({ referencePath: nextPath })}
            />

            <AnimatePresence>
              {busy && (
                <motion.div initial={{ opacity: 0, scale: 0.98 }} animate={{ opacity: 1, scale: 1 }} exit={{ opacity: 0, scale: 0.98 }}>
                  <CreateStatus step={step} status={status} />
                </motion.div>
              )}
            </AnimatePresence>
            {error && <ErrorBlock className="mt-0">{error}</ErrorBlock>}
          </aside>
        </div>
      </div>
    </AppFrame>
  );
}
