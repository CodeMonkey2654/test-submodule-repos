// src/hooks/useKeyboard.ts
import { useEffect, useState } from 'react';

type GameAction = 'moveUp' | 'moveDown' | 'moveLeft' | 'moveRight' | 'drop' | 'dig';

const keyActionMap: { [key: string]: GameAction } = {
  KeyW: 'moveUp',
  KeyS: 'moveDown',
  KeyA: 'moveLeft',
  KeyD: 'moveRight',
  KeyH: 'drop',
  KeyJ: 'dig',
};

export const useKeyboard = (isActive: boolean): Set<GameAction> => {
  const [activeActions, setActiveActions] = useState<Set<GameAction>>(new Set());

  useEffect(() => {
    if (!isActive) return;

    const handleKeyDown = (event: KeyboardEvent) => {
      const action = keyActionMap[event.code];
      if (action) {
        setActiveActions(prev => new Set(prev).add(action));
      }
    };

    const handleKeyUp = (event: KeyboardEvent) => {
      const action = keyActionMap[event.code];
      if (action) {
        setActiveActions(prev => {
          const newSet = new Set(prev);
          newSet.delete(action);
          return newSet;
        });
      }
    };

    window.addEventListener('keydown', handleKeyDown);
    window.addEventListener('keyup', handleKeyUp);

    return () => {
      window.removeEventListener('keydown', handleKeyDown);
      window.removeEventListener('keyup', handleKeyUp);
    };
  }, [isActive]);

  return activeActions;
};
