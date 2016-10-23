function pane () {
     osascript &>/dev/null <<EOF
        tell application "iTerm"
            activate
            tell current session of current window
                split vertically with default profile
                split vertically with default profile
            end tell
        end tell
    EOF
}

