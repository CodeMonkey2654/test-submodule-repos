export interface TelemetryData {
    timestamp: string; // ISO string
    powerLevel: number; // Percentage
    temperature: number; // Celsius
    location: {
      x: number;
      y: number;
    };
    // Add other telemetry fields as needed
  }
  
  export interface Run {
    id: string;
    name: string;
    startTime: string; // ISO string
    endTime: string; // ISO string
  }
  
  export interface ComparativeData {
    runId: string;
    telemetry: TelemetryData[];
  }
  
  export interface FAQ {
    question: string;
    answer: string;
  }
  
  export interface ContactFormData {
    name: string;
    email: string;
    subject: string;
    message: string;
  }