// src/hooks/useGamepad.ts
import { useEffect, useState, useCallback } from 'react';

type GameAction = 'moveUp' | 'moveDown' | 'moveLeft' | 'moveRight' | 'drop' | 'dig';

const gamepadButtonMap: { [index: number]: GameAction } = {
  0: 'drop',
  1: 'dig'
};

export const useGamepad = (isActive: boolean): Set<GameAction> => {
  const [activeActions, setActiveActions] = useState<Set<GameAction>>(new Set());

  const pollGamepads = useCallback(() => {
    const gamepads = navigator.getGamepads ? Array.from(navigator.getGamepads()).filter(gp => gp) as Gamepad[] : [];
    gamepads.forEach(gamepad => {
      // Handle button presses
      gamepad.buttons.forEach((button, index) => {
        const action = gamepadButtonMap[index];
        if (action) {
          if (button.pressed) {
            setActiveActions(prev => new Set(prev).add(action));
          } else {
            setActiveActions(prev => {
              const newSet = new Set(prev);
              newSet.delete(action);
              return newSet;
            });
          }
        }
      });

      // Handle axes for movement
      const threshold = 0.5;
      const xAxis = gamepad.axes[0];
      const yAxis = gamepad.axes[1];

      setActiveActions(prev => {
        const newSet = new Set(prev);
        if (xAxis < -threshold) newSet.add('moveLeft');
        else newSet.delete('moveLeft');

        if (xAxis > threshold) newSet.add('moveRight');
        else newSet.delete('moveRight');

        if (yAxis < -threshold) newSet.add('moveUp');
        else newSet.delete('moveUp');

        if (yAxis > threshold) newSet.add('moveDown');
        else newSet.delete('moveDown');

        return newSet;
      });
    });

    if (isActive) {
      requestAnimationFrame(pollGamepads);
    }
  }, [isActive]);

  useEffect(() => {
    if (isActive) {
      pollGamepads();
    }
  }, [isActive, pollGamepads]);

  return activeActions;
};
