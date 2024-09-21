// src/contexts/UserContext.tsx

import React, { createContext, useState, useEffect } from 'react';
import {jwtDecode} from 'jwt-decode';

interface User {
  name: string;
  role: 'controls' | 'electrical' | 'mechanical';
  admin: boolean;
  // Add other user fields as needed
}

interface UserContextProps {
  user: User | null;
  token: string | null;
  setToken: (token: string | null) => void;
}

export const UserContext = createContext<UserContextProps>({
  user: null,
  token: null,
  setToken: () => {},
});

interface Props {
  children: React.ReactNode;
}

const UserProvider: React.FC<Props> = ({ children }) => {
  const [token, setToken] = useState<string | null>(null);
  const [user, setUser] = useState<User | null>(null);

  useEffect(() => {
    if (token) {
      const decoded: any = jwtDecode(token);
      setUser({ name: decoded.name, role: decoded.role, admin: decoded.admin });
      // Extract and set other user fields as needed
    } else {
      setUser(null);
    }
  }, [token]);

  return (
    <UserContext.Provider value={{ user, token, setToken }}>
      {children}
    </UserContext.Provider>
  );
};

export default UserProvider;
