import React from 'react';

/**
 * Reusable SpokHand Logo Component
 * 
 * @param {string} variant - 'icon' | 'logo' | 'full' - determines which logo to display
 * @param {string} size - size class (e.g., 'h-8 w-8', 'h-16 w-16')
 * @param {string} className - additional CSS classes
 */
export default function Logo({ variant = 'icon', size = 'h-8 w-8', className = '' }) {
  const getLogoSrc = () => {
    switch (variant) {
      case 'logo':
        return '/spokhand-logo.svg';
      case 'full':
        return '/spokhand-logo.svg';
      case 'icon':
      default:
        return '/spokhand-icon.svg';
    }
  };

  return (
    <img 
      src={getLogoSrc()} 
      alt="SpokHand Logo" 
      className={`${size} ${className}`}
    />
  );
}

export function LogoWithText({ variant = 'icon', size = 'h-8 w-8', textSize = 'text-xl', className = '' }) {
  return (
    <div className={`flex items-center gap-2 ${className}`}>
      <Logo variant={variant} size={size} />
      <h1 className={`font-semibold text-gray-900 ${textSize}`}>SPOKHAND SIGNCUT</h1>
    </div>
  );
}

