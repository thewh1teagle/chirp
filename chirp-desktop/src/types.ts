export type ModelBundle = {
  installed: boolean;
  model_path: string;
  codec_path: string;
  model_dir: string;
  version: string;
  url: string;
};

export type RunnerInfo = {
  base_url: string;
};

export type DownloadProgress = {
  downloaded: number;
  total?: number | null;
  progress?: number | null;
  stage: "downloading" | "extracting";
};

export type CreateStep = "idle" | "starting" | "loading" | "creating" | "done";

export type VoicePreset = {
  id: string;
  name: string;
  description: string;
  language: string;
  url: string;
};

export type VoiceCatalog = {
  version: string;
  source: string;
  text: string;
  voices: VoicePreset[];
};

export type DownloadedVoice = {
  path: string;
};
