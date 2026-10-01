const ACTIVE_JOB_KEY = "leazard_active_job";

// In memory only, so the chosen PDF survives sign-in and "Try again" without a re-pick.
export const draft = { file: null, zip: "" };

export const activeJob = {
  get() { return localStorage.getItem(ACTIVE_JOB_KEY); },
  set(id) { localStorage.setItem(ACTIVE_JOB_KEY, id); },
  clear() { localStorage.removeItem(ACTIVE_JOB_KEY); },
};
