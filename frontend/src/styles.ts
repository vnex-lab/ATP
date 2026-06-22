import type { CSSProperties } from 'react'
import { theme } from './theme'

export const inputStyle: CSSProperties = {
  width: '100%',
  background: theme.input,
  border: `1px solid ${theme.inputBorder}`,
  borderRadius: 4,
  padding: '7px 10px',
  color: theme.text,
  fontSize: 13,
  outline: 'none',
  fontFamily: theme.font,
}

export const textareaStyle: CSSProperties = {
  ...inputStyle,
  resize: 'vertical',
  lineHeight: 1.5,
}

export const labelStyle: CSSProperties = {
  fontSize: 12,
  color: theme.textSecondary,
  display: 'block',
  marginBottom: 6,
  fontWeight: 500,
}

export const hintStyle: CSSProperties = {
  color: theme.textSecondary,
  fontSize: 13,
  lineHeight: 1.6,
  margin: '0 0 12px',
}

export const insetBox: CSSProperties = {
  background: theme.panel,
  border: `1px solid ${theme.border}`,
  borderRadius: 6,
  padding: '10px 14px',
}

export const sectionLabel: CSSProperties = {
  fontSize: 11,
  color: theme.textMuted,
  fontWeight: 600,
  textTransform: 'uppercase',
  letterSpacing: '0.06em',
  marginBottom: 4,
}

export function tabStyle(active: boolean): CSSProperties {
  return {
    padding: '7px 14px',
    borderRadius: 4,
    border: `1px solid ${active ? theme.accent : theme.border}`,
    background: active ? theme.accentMuted : theme.panel,
    color: active ? theme.info : theme.textSecondary,
    fontSize: 13,
    cursor: 'pointer',
    fontWeight: active ? 600 : 400,
    fontFamily: theme.font,
  }
}

export function dropZoneStyle(active: boolean): CSSProperties {
  return {
    border: `2px dashed ${active ? theme.accent : theme.border}`,
    borderRadius: 6,
    padding: '36px 20px',
    textAlign: 'center',
    cursor: 'pointer',
    background: active ? theme.accentMuted : theme.panel,
    transition: 'border-color 0.15s, background 0.15s',
  }
}

export const alertColors = {
  success: { bg: theme.successBg, border: theme.success, text: theme.success },
  error: { bg: theme.errorBg, border: theme.error, text: theme.error },
  info: { bg: theme.infoBg, border: theme.info, text: theme.info },
} as const
