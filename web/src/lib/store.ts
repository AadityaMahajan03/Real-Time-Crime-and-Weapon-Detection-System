import { create } from "zustand";
import { Incident, DetectionSettings, Camera } from "./types";
import { MOCK_INCIDENTS, MOCK_CAMERAS } from "./mock-data";
import { api } from "./api";

interface AppState {
  // Incidents & Drawer
  incidents: Incident[];
  addIncident: (incident: Incident) => void;
  loadIncidents: () => Promise<void>;
  selectedIncident: Incident | null;
  setSelectedIncident: (incident: Incident | null) => void;
  updateIncidentAction: (
    incidentId: string,
    action: "CONFIRMED_THREAT" | "MARKED_FALSE_ALARM" | "ESCALATED",
    notes?: string
  ) => Promise<void>;
  filterSeverity: string;
  filterStatus: string;
  filterCamera: string;
  searchQuery: string;
  setFilterSeverity: (val: string) => void;
  setFilterStatus: (val: string) => void;
  setFilterCamera: (val: string) => void;
  setSearchQuery: (val: string) => void;

  // Cameras & Live Streaming
  cameras: Camera[];
  activeCameraId: string;
  setActiveCamera: (id: string) => void;
  isStreaming: boolean;
  setIsStreaming: (val: boolean) => void;
  activeSource: "demo" | "webcam" | "upload" | "rtsp";
  setActiveSource: (val: "demo" | "webcam" | "upload" | "rtsp") => void;
  selectedModel: string;
  setSelectedModel: (val: string) => void;

  // Detection Settings
  detectionSettings: DetectionSettings;
  updateDetectionSettings: (partial: Partial<DetectionSettings>) => void;

  // UI state
  isSidebarCollapsed: boolean;
  toggleSidebar: () => void;
  currentUser: "Operator_Lead" | "Reviewer_Patel";
  setCurrentUser: (user: "Operator_Lead" | "Reviewer_Patel") => void;
}

export const useAppStore = create<AppState>((set, get) => ({
  incidents: MOCK_INCIDENTS,
  addIncident: (incident) =>
    set((state) => {
      // Don't add duplicate ID
      if (state.incidents.some((i) => i.id === incident.id)) return state;
      return { incidents: [incident, ...state.incidents] };
    }),
  loadIncidents: async () => {
    try {
      const serverIncidents = await api.getIncidents();
      if (serverIncidents && serverIncidents.length > 0) {
        set((state) => ({
          incidents: serverIncidents,
          selectedIncident: state.selectedIncident
            ? serverIncidents.find((i) => i.id === state.selectedIncident?.id) || state.selectedIncident
            : null,
        }));
      }
    } catch {
      // Keep existing incidents
    }
  },
  selectedIncident: null,
  setSelectedIncident: (incident) => set({ selectedIncident: incident }),

  updateIncidentAction: async (incidentId, action, notes) => {
    const { currentUser, incidents, selectedIncident } = get();
    try {
      const updated = await api.updateIncidentStatus(incidentId, action, currentUser, notes);
      set({
        incidents: incidents.map((inc) => (inc.id === incidentId ? updated : inc)),
        selectedIncident: selectedIncident?.id === incidentId ? updated : selectedIncident,
      });
    } catch (err) {
      console.error("Failed to update incident:", err);
    }
  },

  filterSeverity: "all",
  filterStatus: "all",
  filterCamera: "all",
  searchQuery: "",
  setFilterSeverity: (filterSeverity) => set({ filterSeverity }),
  setFilterStatus: (filterStatus) => set({ filterStatus }),
  setFilterCamera: (filterCamera) => set({ filterCamera }),
  setSearchQuery: (searchQuery) => set({ searchQuery }),

  cameras: MOCK_CAMERAS,
  activeCameraId: "cam-01",
  setActiveCamera: (id) => set({ activeCameraId: id }),
  isStreaming: true,
  setIsStreaming: (val) => set({ isStreaming: val }),
  activeSource: "demo",
  setActiveSource: (activeSource) => set({ activeSource }),
  selectedModel: "model-yolo-weapon",
  setSelectedModel: (selectedModel) => set({ selectedModel }),

  detectionSettings: {
    weaponThreshold: 0.50,
    violenceThreshold: 0.50,
    fpsLimit: 30,
    sirenEnabled: true,
    boxSmoothing: true,
    alertOnHighOnly: false,
  },
  updateDetectionSettings: (partial) =>
    set((state) => ({
      detectionSettings: { ...state.detectionSettings, ...partial },
    })),

  isSidebarCollapsed: false,
  toggleSidebar: () => set((state) => ({ isSidebarCollapsed: !state.isSidebarCollapsed })),
  currentUser: "Operator_Lead",
  setCurrentUser: (currentUser) => set({ currentUser }),
}));
