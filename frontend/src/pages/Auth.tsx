import React, { useState } from 'react';
import { LoginForm } from '../components/auth/LoginForm';
import { RegisterForm } from '../components/auth/RegisterForm';
import { Card, CardContent } from '../components/ui/card';

export default function Auth() {
  const [isLogin, setIsLogin] = useState(true);

  return (
    <div className="min-h-screen bg-gradient-to-br from-blue-50 via-white to-purple-50 flex items-center justify-center p-4">
      <div className="w-full max-w-md">
        {/* Logo/Brand */}
        <div className="text-center mb-8">
          <div className="flex justify-center items-center gap-3 mb-4">
            <img src="/spokhand-logo.svg" alt="SpokHand Logo" className="h-16 w-16" />
            <h1 className="text-3xl font-bold text-gray-900">
              SPOKHAND SIGNCUT
            </h1>
          </div>
          <p className="text-gray-600">
            Sign Language Dataset Creation Platform
          </p>
        </div>

        {/* Auth Forms */}
        {isLogin ? (
          <LoginForm onSwitchToRegister={() => setIsLogin(false)} />
        ) : (
          <RegisterForm onSwitchToLogin={() => setIsLogin(true)} />
        )}

        {/* Footer */}
        <div className="text-center mt-8 text-sm text-gray-500">
          <p>By continuing, you agree to our Terms of Service and Privacy Policy</p>
        </div>
      </div>
    </div>
  );
} 