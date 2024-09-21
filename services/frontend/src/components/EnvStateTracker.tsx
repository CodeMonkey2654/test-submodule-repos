import { FC, useEffect, useState } from "react";
import { ResponsiveContainer } from "recharts";
import { Graph } from "react-d3-graph";
import { Card, Spinner } from "./index";

interface Transition {
  from: number;
  to: number;
  probability: number;
}

interface EnvStateData {
  states: number[];
  transitions: Transition[];
}

const EnvStateTracker: FC = () => {
  const [envStateData, setEnvStateData] = useState<EnvStateData>({
    states: Array.from({ length: 10 }, (_, index) => index),
    transitions: Array.from({ length: 20 }, () => ({
      from: Math.floor(Math.random() * 10),
      to: Math.floor(Math.random() * 10),
      probability: Math.random(),
    })),
  });

  const [loading, setLoading] = useState<boolean>(true);

  useEffect(() => {
    // Simulate initial data loading
    const loadData = setTimeout(() => {
      setLoading(false);
    }, 1000);

    // Update data every second
    const interval = setInterval(() => {
      setEnvStateData((prevData) => {
        const newStates = prevData.states.map(
          (state) => state + Math.floor(Math.random() * 3) - 1
        );
        const newTransitions = prevData.transitions.map((transition) => {
          const probabilityChange = Math.random() * 0.2 - 0.1;
          const newProbability = Math.max(
            0,
            Math.min(1, transition.probability + probabilityChange)
          );
          return {
            ...transition,
            probability: newProbability,
          };
        });
        return { states: newStates, transitions: newTransitions };
      });
    }, 1000);

    return () => {
      clearInterval(interval);
      clearTimeout(loadData);
    };
  }, []);

  if (loading) {
    return (
      <Card title="Environment State Tracker" className="flex justify-center items-center h-96">
        <Spinner />
      </Card>
    );
  }

  return (
    <Card title="Environment State Tracker" className="w-full h-96">
      <ResponsiveContainer width="100%" height="100%">
        <Graph
          id="env-graph"
          data={{
            nodes: envStateData.states.map((_, index) => ({
              id: `state-${index}`,
              label: `State ${index}`,
            })),
            links: envStateData.transitions.map((transition, index) => ({
              source: `state-${transition.from}`,
              target: `state-${transition.to}`,
              id: `transition-${index}`,
              label: transition.probability.toFixed(2),
            })),
          }}
          config={{
            nodeHighlightBehavior: true,
            node: {
              color: "#DC2626", // Primary color (red)
              size: 300,
              labelProperty: "label",
              fontSize: 14,
              fontColor: "#000000", // Black
              fontWeight: "bold",
            },
            link: {
              color: "#4B5563", // Gray-700
              labelProperty: "label",
              renderLabel: true,
              fontSize: 12,
              fontColor: "#000000", // Black
              fontWeight: "bold",
              strokeWidth: 2,
            },
            d3: {
              gravity: -300,
              linkLength: 120,
              linkStrength: 1,
              alphaTarget: 0.1,
            },
            directed: true,
            automaticRearrangeAfterDropNode: true,
            collapsible: false,
            height: 380,
            width: 600,
            maxZoom: 2,
            minZoom: 0.5,
            panAndZoom: true,
          }}
        />
      </ResponsiveContainer>
    </Card>
  );
};

export default EnvStateTracker;
