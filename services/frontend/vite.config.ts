import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";

// GITHUB_REPOSITORY is set automatically inside GitHub Actions
// ("J-Mango-19/win_predictor_pipeline"), so a Pages build resolves to
// /win_predictor_pipeline/ while local dev stays at "/". App.tsx reads the
// result back via import.meta.env.BASE_URL.
const repositoryName = process.env.GITHUB_REPOSITORY?.split("/")[1];

export default defineConfig({
  base: repositoryName ? `/${repositoryName}/` : "/",
  plugins: [react()],
});
