import React, { createContext, useContext, useEffect, useState } from 'react';
import { User, Session } from '@supabase/supabase-js';
import { supabase } from '@/integrations/supabase/client';
import type { AppRole, Profile, Institution } from '@/types/database';

interface AuthContextType {
  user: User | null;
  session: Session | null;
  profile: Profile | null;
  role: AppRole | null;
  institution: Institution | null;
  loading: boolean;
  signUp: (email: string, password: string, fullName: string, role: AppRole, institutionData?: Partial<Institution>) => Promise<{ error: Error | null }>;
  signIn: (email: string, password: string) => Promise<{ error: Error | null }>;
  signOut: () => Promise<void>;
  refreshUserData: () => Promise<void>;
}

const AuthContext = createContext<AuthContextType | undefined>(undefined);

export function AuthProvider({ children }: { children: React.ReactNode }) {
  const [user, setUser] = useState<User | null>(null);
  const [session, setSession] = useState<Session | null>(null);
  const [profile, setProfile] = useState<Profile | null>(null);
  const [role, setRole] = useState<AppRole | null>(null);
  const [institution, setInstitution] = useState<Institution | null>(null);
  const [loading, setLoading] = useState(true);

  const fetchUserData = async (userId: string) => {
    try {
      // Fetch profile
      const { data: profileData } = await supabase
        .from('profiles')
        .select('*')
        .eq('user_id', userId)
        .maybeSingle();
      
      if (profileData) {
        setProfile(profileData as Profile);
      }

      // Fetch role
      const { data: roleData } = await supabase
        .from('user_roles')
        .select('*')
        .eq('user_id', userId)
        .maybeSingle();
      
      if (roleData) {
        setRole(roleData.role as AppRole);
      }

      // Fetch institution if role is institution
      if (roleData?.role === 'institution') {
        const { data: institutionData } = await supabase
          .from('institutions')
          .select('*')
          .eq('user_id', userId)
          .maybeSingle();
        
        if (institutionData) {
          setInstitution(institutionData as Institution);
        }
      }
    } catch (error) {
      console.error('Error fetching user data:', error);
    }
  };

  const refreshUserData = async () => {
    if (user) {
      await fetchUserData(user.id);
    }
  };

  useEffect(() => {
    // Set up auth state listener FIRST
    const { data: { subscription } } = supabase.auth.onAuthStateChange(
      async (event, session) => {
        setSession(session);
        setUser(session?.user ?? null);
        
        if (session?.user) {
          // Use setTimeout to avoid Supabase deadlock
          setTimeout(() => fetchUserData(session.user.id), 0);
        } else {
          setProfile(null);
          setRole(null);
          setInstitution(null);
        }
        setLoading(false);
      }
    );

    // THEN check for existing session
    supabase.auth.getSession().then(({ data: { session } }) => {
      setSession(session);
      setUser(session?.user ?? null);
      if (session?.user) {
        fetchUserData(session.user.id);
      }
      setLoading(false);
    });

    return () => subscription.unsubscribe();
  }, []);

  const signUp = async (
    email: string,
    password: string,
    fullName: string,
    selectedRole: AppRole,
    institutionData?: Partial<Institution>
  ) => {
    try {
      const { data, error } = await supabase.auth.signUp({
        email,
        password,
        options: {
          emailRedirectTo: window.location.origin,
          data: { full_name: fullName }
        }
      });

      if (error) throw error;

      if (data.user) {
        // Insert role
        const { error: roleError } = await supabase
          .from('user_roles')
          .insert({ user_id: data.user.id, role: selectedRole });

        if (roleError) throw roleError;

        // If institution role, create institution record
        if (selectedRole === 'institution' && institutionData) {
          const { error: instError } = await supabase
            .from('institutions')
            .insert({
              user_id: data.user.id,
              name: institutionData.name || '',
              type: institutionData.type || 'college',
              state: institutionData.state || '',
              district: institutionData.district || '',
              address: institutionData.address,
              establishment_year: institutionData.establishment_year,
            });

          if (instError) throw instError;
        }
      }

      return { error: null };
    } catch (error) {
      return { error: error as Error };
    }
  };

  const signIn = async (email: string, password: string) => {
    try {
      const { error } = await supabase.auth.signInWithPassword({
        email,
        password,
      });

      if (error) throw error;
      return { error: null };
    } catch (error) {
      return { error: error as Error };
    }
  };

  const signOut = async () => {
    await supabase.auth.signOut();
    setUser(null);
    setSession(null);
    setProfile(null);
    setRole(null);
    setInstitution(null);
  };

  return (
    <AuthContext.Provider
      value={{
        user,
        session,
        profile,
        role,
        institution,
        loading,
        signUp,
        signIn,
        signOut,
        refreshUserData,
      }}
    >
      {children}
    </AuthContext.Provider>
  );
}

export function useAuth() {
  const context = useContext(AuthContext);
  if (context === undefined) {
    throw new Error('useAuth must be used within an AuthProvider');
  }
  return context;
}
