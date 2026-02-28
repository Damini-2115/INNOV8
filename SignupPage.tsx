import { Navigate } from 'react-router-dom';
import { useAuth } from '@/contexts/AuthContext';
import { SignupForm } from '@/components/auth/SignupForm';
import { Loader2 } from 'lucide-react';

export default function SignupPage() {
  const { user, role, loading } = useAuth();

  if (loading) {
    return (
      <div className="flex min-h-screen items-center justify-center bg-background">
        <Loader2 className="h-8 w-8 animate-spin text-primary" />
      </div>
    );
  }

  if (user && role) {
    return <Navigate to={role === 'officer' ? '/officer' : '/institution'} replace />;
  }

  return (
    <div className="flex min-h-screen flex-col items-center justify-center bg-gradient-to-br from-primary/5 via-background to-accent/5 p-4">
      <SignupForm />
      <p className="mt-8 text-center text-xs text-muted-foreground">
        Government of India | Ministry of Education
        <br />
        Institutional Compliance Management System
        <br />
        Smart Unified Governance and Approval Management
      </p>
    </div>
  );
}
